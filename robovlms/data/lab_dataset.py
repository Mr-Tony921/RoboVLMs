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
        norm_action=True,
        # traj_per_episode=26,
        # traj_length=10,
        stride=1,
        data_cam_list=None,
        obs_n_frames=1,
        include_target=0,
        out_size=224,
        remove_small_diff=False,
        cache_in_memory=False,
        data_aug=False,
        task_type="lab_action",
        model_name="kosmos",
        **kwargs,
    ):
        self.data_path = data_path
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.traj_per_episode = window_size + fwd_pred_next_n
        self.traj_length = fwd_pred_next_n
        self.norm_action = norm_action
        self.obs_n_frames = obs_n_frames                                                                                                                                 
        self.include_target = include_target
        self.stride = stride
        self.remove_small_diff = remove_small_diff
        self.cache_in_memory = cache_in_memory
        self.task_type = task_type

        self.image_fn = image_fn
        self.tokenizer = tokenizer
        self.text_fn = get_text_function(self.tokenizer, model_name)

        # need to update when change lab dataset
        self.act_q01 = torch.tensor([-0.086, -0.249, -0.178, -0.539, -0.283, -0.468, 0.0])
        self.act_q99 = torch.tensor([0.137, 0.175, 0.164, 0.332, 0.276, 0.553, 1.0])

        print('remove_small_diff', remove_small_diff)

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

        if data_aug:
            self.data_transform1 = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ColorJitter(brightness=0.3, contrast=[0.7, 1.3], saturation=[0.7, 1.3], hue=0.07),
                    torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )
        else:
            self.data_transform1 = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
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
                    else:
                        observation["image"] = np.zeros_like(traj["observation"]['image'][-1])

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

                    if j == self.traj_length - 1 and k != 'image' and k != 'seg':
                        traj["observation"][k] = torch.stack(traj["observation"][k], dim=0)

                if j == self.obs_n_frames - 1 and 'image' in observation.keys():
                    traj["observation"]['image'] = np.stack(traj["observation"]['image'], axis=0)

                    aaa = traj["observation"]['image']
                    tmp_img_inp = np.transpose(aaa, (1,2,0,3)).reshape(aaa.shape[1], aaa.shape[2], aaa.shape[0]*aaa.shape[3])
                    tmp_img_inp = self.data_transform(tmp_img_inp)
                    tmp_img_inp = tmp_img_inp.reshape(aaa.shape[0], aaa.shape[3], tmp_img_inp.shape[1], tmp_img_inp.shape[2])

                    # L C H W
                    t_shape = tmp_img_inp.shape

                    tmp_img_inp = self.data_transform1(tmp_img_inp.permute(1,2,0,3).flatten(2,3))
                    tmp_img_inp = tmp_img_inp.reshape(t_shape[1], t_shape[2], t_shape[0], t_shape[3]).permute(2,0,1,3)
                    traj["observation"]['image'] = tmp_img_inp

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
        action_chunck = torch.stack(
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
        )
        if self.norm_action:
            action_chunck = 2 * (action_chunck - self.act_q01) / (self.act_q99 - self.act_q01) - 1
        action_chunck = action_chunck[:, : self.window_size]

        action_mask = torch.stack([s["action"]["terminate_episode"] for s in sample])
        action_mask = (~torch.all(action_mask == torch.tensor([1, 0, 0]), dim=-1))
        action_mask = action_mask[:, : self.window_size]

        action_tensors = action_chunck[:, :, 0]

        images = torch.stack([s["observation"]["image"].squeeze() for s in sample])
        B, T, C, H, W = images.shape
        image_list = [
            Image.fromarray(images[b, t].permute(1, 2, 0).byte().numpy())
            for b in range(B) for t in range(T)
        ]
        image_tensors = self.image_fn(image_list).view(B, T, C, H, W)
        image_chunk = generate_chunck_data(image_tensors, self.window_size, self.fwd_pred_next_n)
        fwd_mask = action_mask
        image_tensors = image_tensors[:, : self.window_size]

        gripper_tensors = None
        gripper_chunk = None

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
        data_path="/mnt/afs/share_data/duanhaonan/datasets/lab_dataset/LabData_L1_807",
        image_fn=image_fn,
        tokenizer=model.model.tokenizer,
        window_size=16,
        fwd_pred_next_n=10,
        stride=4,
        include_target=1,
        obs_n_frames=1,
        remove_small_diff=True,
        cache_in_memory=True,
        data_aug=True,
        norm_action=False,
    )    

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=dataset.collater,
        drop_last=True,
        shuffle=True,
        num_workers=8,
    )

    total_iter_num = 0
    action_list = []
    for ii in range(100):
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

def main_ori():
    print("begin!", flush=True)
    tokenizer_config = {
        "type": "AutoProcessor",
        "pretrained_model_name_or_path": "microsoft/kosmos-2-patch14-224",
        "tokenizer_type": "kosmos",
        "max_text_len": 256,
    }
    tokenizer = build_tokenizer(tokenizer_config)
    dataset = LabDataset(
        data_path="/mnt/afs/share_data/duanhaonan/datasets/lab_dataset/LabData_L1_807",
        tokenizer=tokenizer,
        traj_per_episode=16,
        traj_length=10,
        stride=4,
        include_target=1,
        obs_n_frames=1,
        remove_small_diff=True,
        cache_in_memory=True,
    )    

    wv_min = torch.ones(3) * 1000
    wv_max = torch.ones(3) * -1000
    rt_min = torch.ones(4) * 1000
    rt_max = torch.ones(4) * -1000
    pose_min = torch.ones(6) * 1000
    pose_max = torch.ones(6) * -1000
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)

    total_iter_num = 0
    action_list = []
    proprio_list = []
    for ii in range(100):
        for i, batch in enumerate(dataloader):
            import ipdb;ipdb.set_trace()
            proprio_list.append(torch.cat([batch['action']['state_pose'], 
                                           batch["action"]["gripper_closedness_action"]], dim=-1).flatten(0, 2).cpu().numpy())
            action_list.append(torch.cat([batch["action"]["world_vector"].flatten(0, 2), batch["action"]["rotation_delta"].flatten(0, 2),
                                          batch["action"]["gripper_closedness_action"].flatten(0, 2)], dim=-1).cpu().numpy())
            wv_min = torch.minimum(
                    wv_min, batch["action"]["world_vector"].amin(dim=(0, 1, 2))
                )
            wv_max = torch.maximum(
                wv_max, batch["action"]["world_vector"].amax(dim=(0, 1, 2))
                )
            # rt_min = torch.minimum(
            #     rt_min, batch["action"]["rotation_delta"].amin(dim=(0, 1, 2))
            #     )
            # rt_max = torch.maximum(
            #     rt_max, batch["action"]["rotation_delta"].amax(dim=(0, 1, 2))
            #     )
            if total_iter_num % 50 == 0:

                print('act min', np.min(np.concatenate(action_list), axis=0).tolist(), flush=True)
                print('act max', np.max(np.concatenate(action_list), axis=0).tolist(), flush=True)
                print('act low', np.quantile(np.concatenate(action_list), 0.01, axis=0).tolist(), flush=True)
                print('act high', np.quantile(np.concatenate(action_list), 0.99, axis=0).tolist(), flush=True)
                print('act mean', np.mean(np.concatenate(action_list), axis=0).tolist(), flush=True)
                print('act std', np.std(np.concatenate(action_list), axis=0).tolist(), flush=True)

                print('prio min', np.min(np.concatenate(proprio_list), axis=0).tolist(), flush=True)
                print('prio max', np.max(np.concatenate(proprio_list), axis=0).tolist(), flush=True)
                print('prio low', np.quantile(np.concatenate(proprio_list), 0.01, axis=0).tolist(), flush=True)
                print('prio high', np.quantile(np.concatenate(proprio_list), 0.99, axis=0).tolist(), flush=True)
                print('prio mean', np.mean(np.concatenate(proprio_list), axis=0).tolist(), flush=True)
                print('prio std', np.std(np.concatenate(proprio_list), axis=0).tolist(), flush=True)
                
                print("wv_min: ", wv_min, flush = True)
                print("wv_max: ", wv_max, flush = True)
                # print("rt_min: ", rt_min, flush = True)
                # print("rt_max: ", rt_max, flush = True)

            total_iter_num += 1

    print("wv_min: ", wv_min)
    print("wv_max: ", wv_max)
    # print("rt_min: ", rt_min)
    # print("rt_max: ", rt_max)
    print("finish!", flush=True)

if __name__ == "__main__":
    # main_ori()
    main()
