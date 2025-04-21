import os
import pickle
import random
from collections import defaultdict
import numpy as np

from PIL import Image

from pytorch3d.transforms import (
    Transform3d,
    matrix_to_quaternion,
    quaternion_to_matrix
)

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)

import torch
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from dataclasses import dataclass
from multiprocessing import Value

import robovlms
from robovlms.utils.model_utils import build_tokenizer
from robovlms.data.data_utils import (
    generate_chunck_data,
    get_text_function,
    preprocess_image,
)

from typing import Callable

# need to update when change lab dataset
act_q01 = torch.tensor([-0.08000017702579498,
            -0.20688602328300476,
            -0.16130231320858002,
            -0.5491514801979065,
            -0.2616570293903351,
            -0.44785112142562866,
            0.0])
act_q99 = torch.tensor([0.12800641357898712,
            0.17082400619983673,
            0.1557823121547699,
            0.3282812535762787,
            0.2638643980026245,
            0.5106926560401917,
            1.0])

@torch.no_grad()
def get_pose_cam(world2cam, pose1):
    rot_mat1 = quaternion_to_matrix(pose1[3:])
    pose1_mat = torch.eye(4)

    pose1_mat[:3, :3] = rot_mat1
    pose1_mat[:3, 3] = pose1[:3]

    pose1_transform = Transform3d(matrix=pose1_mat.T)

    world2cam_transform = Transform3d(matrix=world2cam.T)
    pose1_cam = pose1_transform.compose(world2cam_transform)
    vector = pose1_cam.get_matrix()[0, -1, :3]
    rotation = matrix_to_quaternion(pose1_cam.get_matrix()[0, :3, :3].T)
    return torch.cat([vector, rotation])

def quaternion_to_euler_radians(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), w**2 + z**2 - (x**2 + y**2))

    sinpitch = 2 * (w * y - z * x)
    pitch = np.arcsin(sinpitch)

    yaw = np.arctan2(2 * (w * z + x * y), w**2 + x**2 - (y**2 + z**2))

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)

def unnormalize(x):
    x = x.clone()
    for i in range(3):
        x[..., i] = x[..., i] * IMAGENET_DEFAULT_STD[i] + IMAGENET_DEFAULT_MEAN[i]

    return x

@torch.no_grad()
def process_traj(world2cam, pose1, pose2):
    rot_mat1 = quaternion_to_matrix(pose1[3:])
    rot_mat2 = quaternion_to_matrix(pose2[3:])
    pose1_mat, pose2_mat = torch.eye(4), torch.eye(4)

    pose1_mat[:3, :3] = rot_mat1
    pose2_mat[:3, :3] = rot_mat2
    pose1_mat[:3, 3] = pose1[:3]
    pose2_mat[:3, 3] = pose2[:3]

    pose1_transform = Transform3d(matrix=pose1_mat.T)
    pose2_transform = Transform3d(matrix=pose2_mat.T)
    world2cam_transform = Transform3d(matrix=world2cam.T)
    pose1_cam = pose1_transform.compose(world2cam_transform)
    pose2_cam = pose2_transform.compose(world2cam_transform)

    pose1_to_pose2 = pose1_cam.inverse().compose(pose2_cam)

    # translation_delta = pose1_to_pose2.get_matrix()[0, -1, :3]
    translation_delta = (
        pose2_cam.get_matrix()[0, -1, :3] - pose1_cam.get_matrix()[0, -1, :3]
    )

    rotation_delta = matrix_to_quaternion(pose1_to_pose2.get_matrix()[0, :3, :3].T)

    return translation_delta.to(torch.float32), rotation_delta.to(torch.float32)

def is_diff_small(pose1, pose2, threshold_sum=2e-2, threshold_max=5e-3):
    diff_sum = abs(np.asarray(pose2 - pose1)).sum()
    diff_max = abs(np.asarray(pose2 - pose1)).max()
    if diff_sum <= threshold_sum and diff_max <= threshold_max:
        return True
    else:
        return False

class LabDataset(Dataset):

    def __init__(
        self,
        image_fn: Callable,
        tokenizer: Callable,
        data_path,
        window_size=16,
        fwd_pred_next_n=10,
        norm=True,
        traj_per_episode=1,
        # traj_length=10, # fwd_pred_next_n + window_size
        stride=1,
        data_cam_list=None,
        # obs_n_frames=1, # window_size
        include_target=0,
        out_size=224,
        remove_small_diff=False,
        cache_in_memory=False,
        task_type="lab_action",
        model_name="kosmos",
        is_training=True,
        **kwargs,
    ):
        self.data_path = data_path
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.traj_per_episode = traj_per_episode
        self.traj_length = fwd_pred_next_n + window_size
        self.norm = norm
        self.obs_n_frames = self.traj_length

        self.include_target = include_target
        self.stride = stride
        self.remove_small_diff = remove_small_diff
        self.cache_in_memory = cache_in_memory
        self.task_type = task_type
        self.is_training = is_training

        self.image_fn = image_fn
        self.tokenizer = tokenizer
        self.text_fn = get_text_function(self.tokenizer, model_name)

        print('data_path', self.data_path)
        print('window_size', self.window_size)
        print('fwd_pred_next_n', self.fwd_pred_next_n)
        print('traj_per_episode', self.traj_per_episode)
        print('norm', self.norm)
        print('include_target', self.include_target)
        print('stride', self.stride)
        print('remove_small_diff', self.remove_small_diff)
        print('is_training', self.is_training)

        if data_cam_list:
            self.data_cam_list = pickle.load(open(data_cam_list, "rb"))
        else:
            self.data_cam_list = sorted(os.listdir(self.data_path))

        self.data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),                      
                torchvision.transforms.CenterCrop(size=(480, 480)),
                torchvision.transforms.Resize((out_size, out_size), antialias=True)
            ]
        )

        self.cache_data = {}
        # self._preload_data_into_cache()

    def _preload_data_into_cache(self):

        for i in range(len(self.data_cam_list)):

            data_pkl = self.get_cached_data_pkl(i)
            if i % 10 == 0:
                print(f"PRELOAD DATA {i} done!", flush = True)

    def __len__(self):
        if self.is_training:
            return len(self.data_cam_list) * 100
        return len(self.data_cam_list)

    @torch.no_grad()
    def construct_traj(self, episode, episode_path):
        stride = self.stride

        gripper_closeness = np.array([episode["steps"][_]["observation"]["gripper_position"] for _ in range(len(episode["steps"]))])
        gripper_change = np.where(gripper_closeness[1:] != gripper_closeness[:-1])[0]
        gripper_change = np.concatenate([gripper_change, gripper_change + 1])
        gripper_change.sort()

        episode_step = []
        start = random.randint(0, min(stride, len(episode["steps"])) - 1)
        for i in range(len(gripper_change)):
            episode_step.extend(episode["steps"][start : gripper_change[i] : stride])
            start = gripper_change[i]

        episode_step.extend(episode["steps"][start::stride])
        episode["steps"] = episode_step

        steps = len(episode["steps"])
        start_frame = np.random.permutation(steps)[: self.traj_per_episode]

        if self.include_target:
            start_frame = np.random.permutation(steps)[: self.traj_per_episode] - self.obs_n_frames + 1
        if self.include_target == 2: # only for the approaching stage.
            start_frame = np.random.permutation(np.arange(max(steps - 2*1, 1), steps-1, 1))
        if len(start_frame) < self.traj_per_episode:
            start_frame = np.random.choice(start_frame, self.traj_per_episode, replace=True)

        if not self.is_training:
            start_frame = np.concatenate([np.arange(steps),np.arange(steps),np.arange(steps),np.arange(steps),np.arange(steps),np.arange(steps),np.arange(steps),np.arange(steps),np.arange(steps),np.arange(steps),np.arange(steps),np.arange(steps)], axis=0)[:self.traj_per_episode] - self.obs_n_frames + 1

        gripper_change_list = [0]
        for i in range(1, steps):
            if episode['steps'][i]["observation"]["gripper_position"] != episode['steps'][i-1]["observation"]["gripper_position"]:
                gripper_change_list.append(1)
            else:
                gripper_change_list.append(0)
        next_gripper_change_position_dict = {}
        has_changed_gripper_step = False
        tmp_gripper_change_position = steps - 1
        for i in range(steps-1, -1, -1):
            next_gripper_change_position_dict[i] = tmp_gripper_change_position
            if gripper_change_list[i] and (i+1 >=steps or i+1 < steps and not gripper_change_list[i+1]):
                # we use the last change step
                has_changed_gripper_step = True
                tmp_gripper_change_position = i
        has_changed_gripper_step = False

        trajs = {"observation": defaultdict(list), "action": defaultdict(list)}

        for i in range(self.traj_per_episode):
            frame_idx = start_frame[i]
            traj = {"observation": defaultdict(list), "action": defaultdict(list)}

            for j in range(self.traj_length):
                current_frame_idx = max(frame_idx + j, 0)  # this is for the case when frame_idx < 0
                observation = {}
                action = {}

                observation['current_frame_idx'] = torch.tensor(current_frame_idx)

                if j < self.obs_n_frames:
                    if current_frame_idx < steps:
                        observation["image"] = episode["steps"][current_frame_idx]["observation"]["exterior_image_1_left"]
                        observation["wrist"] = episode["steps"][current_frame_idx]["observation"]["exterior_image_1_wrist"]
                    else:
                        observation["image"] = np.zeros_like(traj["observation"]['image'][-1])
                        observation["wrist"] = np.zeros_like(traj["observation"]['wrist'][-1])

                if current_frame_idx == steps - 1:
                    action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor(
                        1. if episode["steps"][current_frame_idx]["observation"]["gripper_position"] > 0.2 else 0.0,
                        dtype=torch.float32,
                    ).unsqueeze(-1)
                elif current_frame_idx >  steps - 1:
                    action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor(
                        0.0,
                        dtype=torch.float32,
                    ).unsqueeze(-1)

                else:
                    action["terminate_episode"] = torch.tensor([0, 1, 0], dtype=torch.int32)
                    action["gripper_closedness_action"] = torch.tensor(
                        1. if episode["steps"][current_frame_idx + 1]["observation"]["gripper_position"] > 0.2 else 0.0,
                        dtype=torch.float32,
                    ).unsqueeze(-1)

                action["loss_weight"] = torch.ones((9))

                if current_frame_idx < steps - 1:
                    pose1 = torch.tensor(episode["steps"][current_frame_idx]["observation"]['cartesian_position']).clone()
                    pose2 = torch.tensor(episode["steps"][current_frame_idx + 1]["observation"]['cartesian_position']).clone()

                    action["world_vector"], action["rotation_delta"] = process_traj(
                        torch.eye(4),
                        pose1.to(torch.float32),
                        pose2.to(torch.float32),
                    )

                    action["rotation_delta"] = quaternion_to_euler_radians(action['rotation_delta'][0], action['rotation_delta'][1], action['rotation_delta'][2], action['rotation_delta'][3]) 
                    action['abs_tar_pose'] = get_pose_cam(torch.eye(4), pose2)
                    action['state_pose'] = get_pose_cam(torch.eye(4), pose1)

                    tmp_abs_pose =  action['abs_tar_pose'][:6].clone()
                    tmp_abs_pose[3:] = quaternion_to_euler_radians( action['abs_tar_pose'][3],  action['abs_tar_pose'][4],  action['abs_tar_pose'][5],  action['abs_tar_pose'][6]) 
                    action['abs_tar_pose'] = tmp_abs_pose

                    tmp_abs_pose =  action['state_pose'][:6].clone()
                    tmp_abs_pose[3:] = quaternion_to_euler_radians( action['state_pose'][3],  action['state_pose'][4],  action['state_pose'][5],  action['state_pose'][6]) 
                    action['state_pose'] = tmp_abs_pose
                else:
                    action["loss_weight"] = torch.zeros((9))
                    action["world_vector"] = torch.zeros(3)
                    action['rotation_delta'] = torch.zeros(3)
                    tmp_pose = torch.tensor(episode['steps'][-1]["observation"]['cartesian_position']).clone()
                    action['abs_tar_pose'] =  get_pose_cam(torch.eye(4), tmp_pose)
                    action['state_pose'] =  get_pose_cam(torch.eye(4), tmp_pose)

                    tmp_abs_pose =  action['abs_tar_pose'][:6].clone()
                    tmp_abs_pose[3:] = quaternion_to_euler_radians( action['abs_tar_pose'][3],  action['abs_tar_pose'][4],  action['abs_tar_pose'][5],  action['abs_tar_pose'][6]) 
                    action['abs_tar_pose'] = tmp_abs_pose

                    tmp_abs_pose =  action['state_pose'][:6].clone()
                    tmp_abs_pose[3:] = quaternion_to_euler_radians( action['state_pose'][3],  action['state_pose'][4],  action['state_pose'][5],  action['state_pose'][6]) 
                    action['state_pose'] = tmp_abs_pose

                if (
                    current_frame_idx > 0
                    and current_frame_idx < steps
                    and episode["steps"][current_frame_idx]["observation"][
                        "gripper_position"
                    ]
                    != episode["steps"][current_frame_idx - 1]["observation"][
                        "gripper_position"
                    ]
                ):
                    action["loss_weight"][7] = 100.0
                if (
                    current_frame_idx > 1 and current_frame_idx < steps
                    and episode["steps"][current_frame_idx]["observation"]["gripper_position"] != episode["steps"][current_frame_idx - 2]["observation"]["gripper_position"]
                ):
                    action["loss_weight"][7] = 100.0

                for k in observation.keys():
                    traj["observation"][k].append(observation[k])

                    if j == self.traj_length - 1 and k != 'image' and k != 'wrist' and k != 'seg':
                        traj["observation"][k] = torch.stack(traj["observation"][k], dim=0)

                if j == self.obs_n_frames - 1 and 'image' in observation.keys():
                    traj["observation"]['image'] = np.stack(traj["observation"]['image'], axis=0)

                    aaa = traj["observation"]['image']
                    tmp_img_inp = np.transpose(aaa, (1,2,0,3)).reshape(aaa.shape[1], aaa.shape[2], aaa.shape[0]*aaa.shape[3])
                    tmp_img_inp = self.data_transform(tmp_img_inp)
                    tmp_img_inp = tmp_img_inp.reshape(aaa.shape[0], aaa.shape[3], tmp_img_inp.shape[1], tmp_img_inp.shape[2])
                    traj["observation"]['image'] = tmp_img_inp

                if j == self.obs_n_frames - 1 and 'wrist' in observation.keys():
                    traj["observation"]['wrist'] = np.stack(traj["observation"]['wrist'], axis=0)

                    aaa = traj["observation"]['wrist']
                    tmp_img_inp = np.transpose(aaa, (1,2,0,3)).reshape(aaa.shape[1], aaa.shape[2], aaa.shape[0]*aaa.shape[3])
                    tmp_img_inp = self.data_transform(tmp_img_inp)
                    tmp_img_inp = tmp_img_inp.reshape(aaa.shape[0], aaa.shape[3], tmp_img_inp.shape[1], tmp_img_inp.shape[2])
                    traj["observation"]['wrist'] = tmp_img_inp

                for k in action.keys():
                    traj["action"][k].append(action[k])
                    if j == self.traj_length - 1:
                        traj["action"][k] = torch.stack(traj["action"][k], dim=0)

            if has_changed_gripper_step:
                # the target position of current observation
                # we use the gripper change step idx to indicate
                # frame_idx is the start index
                target_position_step = next_gripper_change_position_dict[frame_idx+(self.obs_n_frames - 1)*1]
                target_position_pose = torch.tensor(episode['steps'][target_position_step]['observation']['cartesian_position']).clone()

                gripper_change_pose = get_pose_cam(torch.eye(4), target_position_pose)

                t_gripper_position = torch.tensor([episode["steps"][target_position_step]["observation"]["gripper_position"]], dtype=torch.float32)
                if target_position_step == steps - 1:
                    t_terminate_episode = torch.tensor([1, 0, 0], dtype=torch.int32)
                else:
                    t_terminate_episode = torch.tensor([0, 1, 0], dtype=torch.int32)
                gripper_change_pose = torch.cat([gripper_change_pose, t_gripper_position, t_terminate_episode], dim=-1)
            else:
                gripper_change_pose = torch.zeros(11).to(torch.float32) # indicate no target position or we dont know

            trajs["action"]['gripper_change_pose'].append(gripper_change_pose)
            if i == self.traj_per_episode - 1:
                trajs["action"]['gripper_change_pose'] = torch.stack(trajs["action"]['gripper_change_pose'], dim=0)
            for k in traj["observation"].keys():

                trajs["observation"][k].append(traj["observation"][k])
                if i == self.traj_per_episode - 1 and k != 'seg':
                    trajs["observation"][k] = torch.stack(trajs["observation"][k], dim=0)

            for k in traj["action"].keys():
                trajs["action"][k].append(traj["action"][k])
                if i == self.traj_per_episode - 1:
                    trajs["action"][k] = torch.stack(trajs["action"][k], dim=0)
        if trajs is not None:
            trajs['instruction'] = os.path.basename(episode_path).split('_')[1][:-4]
        trajs['ep_path'] = episode_path
        # print('inner', trajs['observation']['image'].min(), trajs['observation']['image'].max())
        return trajs

    @torch.no_grad()
    def __getitem__(self, index):

        index = index % (len(self.data_cam_list))

        while True:

            try:
                data_pkl = self.get_cached_data_pkl(index)
                trajs = self.construct_traj(data_pkl, self.data_cam_list[index])
                break

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                print(f"Fail to load data {self.data_cam_list[index]}", flush = True)
                index = random.randint(0, len(self.data_cam_list)-1)

        return trajs

    def get_cached_data_pkl(self, index):
        if index not in self.cache_data or not self.cache_in_memory:
            data_url = os.path.join(self.data_path, self.data_cam_list[index])

            data_pkl = pickle.load(open(data_url, 'rb'))

            new_steps = []

            if self.remove_small_diff:
                cur =  np.asarray(data_pkl['steps'][0]['observation']['cartesian_position'])
                new_steps = [data_pkl['steps'][0]]
                ii = 1
                while True:
                    temp_step =  np.asarray(data_pkl['steps'][ii]['observation']['cartesian_position'])
                    if ii == len(data_pkl['steps']) - 1:
                        new_steps.append(data_pkl['steps'][ii])
                        break
                    if not is_diff_small(cur, temp_step):
                        new_steps.append(data_pkl['steps'][ii])
                        cur = temp_step
                    ii += 1

                data_pkl['steps'] = new_steps

            if self.cache_in_memory:
                self.cache_data[index] = data_pkl
            # print('no cache', flush = True)
        else:
            import copy
            data_pkl = copy.deepcopy(self.cache_data[index])
            # print('read from cache', flush = True)
        return data_pkl

    def collater(self, sample):
        action_tensors = torch.stack(
            [
                torch.cat(
                    [
                        s["action"]["world_vector"],
                        s["action"]["rotation_delta"],
                        s["action"]["gripper_closedness_action"],
                    ],
                    dim=-1,
                )
                for s in sample
            ]
        )  # (4, 1, 26, 7) (bs, traj_per_episode, fwd_pred_next_n + window_size, action_dim)
        if self.norm:
            action_tensors = torch.clamp(action_tensors, min=act_q01, max=act_q99)
            action_tensors = 2 * (action_tensors - act_q01) / (act_q99 - act_q01) - 1
        action_tensors = action_tensors.squeeze(1)  # (4, 26, 7)
        action_chunck = generate_chunck_data(action_tensors, self.window_size, self.fwd_pred_next_n) # (4, 16, 10, 7)
        action_tensors = action_tensors[:, : self.window_size] # (4, 16, 7)

        action_mask = torch.stack([s["action"]["terminate_episode"] for s in sample]) # (4, 1, 26, 3)
        action_mask = action_mask.squeeze(1)  # (4, 26, 3)
        action_mask = (~torch.all(action_mask == torch.tensor([1, 0, 0]), dim=-1)) # (4, 26)
        action_mask = generate_chunck_data(action_mask, self.window_size, self.fwd_pred_next_n) # (4, 16, 10)

        images = torch.stack([s["observation"]["image"] for s in sample]) # (4, 1, 26, 3, 224, 224)
        images = images.squeeze(1) # (4, 26, 3, 224, 224)
        B, T, C, H, W = images.shape
        image_list = [
            Image.fromarray((images[b, t] * 255).clamp(0, 255).permute(1, 2, 0).byte().numpy())
            for b in range(B) for t in range(T)
        ]
        image_tensors = self.image_fn(image_list).view(B, T, C, H, W)
        image_chunk = generate_chunck_data(image_tensors, self.window_size, self.fwd_pred_next_n)
        fwd_mask = action_mask
        image_tensors = image_tensors[:, : self.window_size]

        wrists = torch.stack([s["observation"]["wrist"] for s in sample]) # (4, 1, 26, 3, 224, 224)
        wrists = wrists.squeeze(1) # (4, 26, 3, 224, 224)
        B, T, C, H, W = wrists.shape
        wrist_list = [
            Image.fromarray((wrists[b, t] * 255).clamp(0, 255).permute(1, 2, 0).byte().numpy())
            for b in range(B) for t in range(T)
        ]
        gripper_tensors = self.image_fn(wrist_list).view(B, T, C, H, W)
        gripper_chunk = generate_chunck_data(gripper_tensors, self.window_size, self.fwd_pred_next_n)
        gripper_tensors = image_tensors[:, : self.window_size]

        stacked_language = [f"In: What action should the robot take to {s['instruction'].lower()}?\nOut:" for s in sample]
        text_tensors, attention_mask = self.text_fn(stacked_language)

        instr_and_action_ids = None
        instr_and_action_labels = None
        instr_and_action_mask = None

        res = {
            "rgb": image_tensors,
            "hand_rgb": gripper_tensors,
            "action": action_tensors,
            "text": text_tensors,
            "text_mask": attention_mask,
            "fwd_rgb_chunck": image_chunk,
            "fwd_hand_rgb_chunck": gripper_chunk,
            "fwd_mask": fwd_mask,
            "action_chunck": action_chunck,
            "chunck_mask": action_mask,
            "instr_and_action_ids": instr_and_action_ids,
            "instr_and_action_labels": instr_and_action_labels,
            "instr_and_action_mask": instr_and_action_mask,
            "raw_text": stacked_language,
            "data_source": self.task_type,
        }
        return res

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def main():
    print("begin!", flush=True)
    torch.set_printoptions(precision=6, sci_mode=False)
    np.set_printoptions(precision=6, suppress=True)

    import json, functools
    from robovlms.train.base_trainer import BaseTrainer
    configs = json.load(open('configs/kosmos_ph_oxe-pretrain.json', 'r'))
    pretrained_path = 'checkpoints/kosmos_ph_oxe-pretrain.pt'
    configs['model_load_path'] = pretrained_path

    model = BaseTrainer.from_checkpoint(pretrained_path, configs.get("model_load_source", "torch"), configs)

    image_fn = functools.partial(
        preprocess_image,
        image_processor=model.model.image_processor,
        model_type=configs["model"],
    )

    dataset = LabDataset(
        data_path="/mnt/afs/share_data/xuyuan2/nips_pkl",
        image_fn=image_fn,
        tokenizer=model.model.tokenizer,
        window_size=16,
        fwd_pred_next_n=10,
        stride=4,
        include_target=0,
        remove_small_diff=True,
        cache_in_memory=True,
        norm=False,
        traj_per_episode=1,
        is_training=True,
    )    

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=dataset.collater,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )

    total_iter_num = 0
    action_list = []
    for ii in range(200):
        for i, batch in enumerate(dataloader):
            action_list.append(batch['action'].flatten(0,-2).cpu().numpy())

            if total_iter_num % 50 == 0:
                print('act min', np.min(np.concatenate(action_list), axis=0).tolist(), flush=True)
                print('act max', np.max(np.concatenate(action_list), axis=0).tolist(), flush=True)
                print('act low', np.quantile(np.concatenate(action_list), 0.01, axis=0).tolist(), flush=True)
                print('act high', np.quantile(np.concatenate(action_list), 0.99, axis=0).tolist(), flush=True)
                print('act mean', np.mean(np.concatenate(action_list), axis=0).tolist(), flush=True)
                print('act std', np.std(np.concatenate(action_list), axis=0).tolist(), flush=True)

            total_iter_num += 1

    print("finish!", flush=True)

if __name__ == "__main__":
    main()
