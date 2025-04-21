"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""
import os
import os.path
import pickle

# import hydra

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import scipy.spatial.transform as st
import torch
import numpy as np
# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


@torch.no_grad()
def euler_to_quaternion(eulers):
    quaternion = st.Rotation.from_euler('xyz', eulers).as_quat()
    return torch.tensor([quaternion[-1], quaternion[0], quaternion[1], quaternion[2]])


# from openvla_warp.LabDataset_openvlaoft import process_traj


# def obtain_gt_delta(actions_gt, step_cnt, camera_extrinsic_cv, stride):
#     step = actions_gt[step_cnt]
#     next_frame_idx = step_cnt + stride
#     next_step = actions_gt[next_frame_idx]
#             # action['world_vector'] = step["action_dict"]['cartesian_position'][:3] - step['observation']['cartesian_position'][:3]
#             # action['rotation_delta'] = step["action_dict"]['cartesian_position'][3:6] - step['observation']['cartesian_position'][3:6]

#     pose1 = torch.tensor(step['observation']['cartesian_position']).clone()
#     pose2 = torch.tensor(next_step["observation"]['cartesian_position']).clone()

#     # pose1 = torch.cat((pose1[:3], pose1[3:]))
#     # pose2 = torch.cat((pose2[:3], pose2[3:]))

#             # pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
#             # pose2 = torch.tensor(base_episode["step"][current_frame_idx ]["target_ee_pose"]).clone()
#             # pose1[0] -= 0.615  # base to world
#             # pose2[0] -= 0.615  # base to world
#     world_vector, rotation_delta = process_traj(
#                 camera_extrinsic_cv if camera_extrinsic_cv is not None else torch.eye(4),
#                 pose1,
#                 pose2,
#             )
#     # rotation_delta*= torch.sign(rotation_delta[0])
#     # rotation_delta[0] -= 1.0
#     world_vector = world_vector.cpu().numpy()
#     rotation_delta = rotation_delta
#     gripper_type = 'next_observation'
#     if gripper_type == 'current_action':
#         gripper_closedness_action = torch.tensor(step['observation']["gripper_position"], dtype=torch.float32)
#     else:
#         gripper_closedness_action = torch.tensor(next_step["observation"]["gripper_position"], dtype=torch.float32)
#     return world_vector,rotation_delta, gripper_closedness_action, pose2

def quaternion_to_euler_radians(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), w**2 + z**2 - (x**2 + y**2))

    sinpitch = 2 * (w * y - z * x)
    pitch = np.arcsin(sinpitch)

    yaw = np.arctan2(2 * (w * z + x * y), w**2 + x**2 - (y**2 + z**2))

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)

def is_diff_small(pose1, pose2, threshold_sum=2e-2, threshold_max=5e-3):
    diff_sum = abs(np.asarray(pose2 - pose1)).sum()
    diff_max = abs(np.asarray(pose2 - pose1)).max()
    if diff_sum <= threshold_sum and diff_max <= threshold_max:
        return True
    else:
        return False

step_cnt = 6
def client_example():
    import requests
    import json_numpy
    json_numpy.patch()
    import numpy as np

    pkl_path = "/mnt/afs/share_data/xuyuan2/nips_pkl/20250409170732_pick up the carrot into the the box.pkl"
    data_pkl = pickle.load(open(pkl_path, 'rb'))

    frames = []
    wrists = []
    for step in data_pkl["steps"]:
        img = step["observation"]["exterior_image_1_left"]
        frames.append(img)
        wrist = step["observation"]["exterior_image_1_wrist"]
        wrists.append(wrist)
    prompt = "pick up the carrot into the the box"
    url_ = "http://0.0.0.0:8777/act"

    for index in range(len(frames)):
        image = frames[index]
        wrist = wrists[index]
        action = requests.post(url_, json={"image": image, "wrist": wrist, "instruction": prompt}).json()
        print(action)

    return

    inp_dir = '/mnt/afs/share_data/xuyuan2/nips_pkl'
    #inp_dir = '/mnt/petrelfs/share_data/zhangtianyi1/Dataset/records_banana_fixgripper/'
    item_list = sorted(os.listdir(inp_dir))
    import random
    ni = random.randint(0, len(item_list))
    ni = 1

    # print(item_list)
    xxx = []
    for ni in range(300):
        actions_gt = pickle.load(open(inp_dir + "/{}".format(item_list[ni]), 'rb'))['steps']
        
        new_steps = []
        if True:
            cur =  np.asarray(actions_gt[0]['observation']['cartesian_position'])
            new_steps = [actions_gt[0]]
            ii = 1
            while True:
                temp_step =  np.asarray(actions_gt[ii]['observation']['cartesian_position'])
                if ii == len(actions_gt) - 1:
                    new_steps.append(actions_gt[ii])
                    break
                if not is_diff_small(cur, temp_step): # 去除小diff
                    new_steps.append(actions_gt[ii])
                    cur = temp_step
                ii += 1

            actions_gt = new_steps

        for step_cnt in range(0, len(actions_gt) -4 ):
            import ipdb; ipdb.set_trace()
            rgb_image_inp1 = actions_gt[step_cnt]["observation"]["exterior_image_1_left"]
            # rgb_image_resized = rgb_image_inp1

            # 640 * 480
            # rgb_image_inp = rgb_image_inp1[:, 80:-80]
            import cv2
            # rgb_image_resized = cv2.resize(rgb_image_inp, (224, 224))
            stride = 4
            import torchvision
            data_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # v2.RandomResizedCrop(size=(224, 224), antialias=True),

                    torchvision.transforms.CenterCrop(size=(480, 480)),
                    torchvision.transforms.Resize((256 , 256), antialias=True)
                    # torchvision.transforms.RandomHorizontalFlip(p=0.5),

                    # torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                    # torchvision.transforms.Pad(padding = (80,1,80,0)),
                    # torchvision.transforms.Resize((224,224), antialias=True)
                ]
            )  


            # instruction = "pick up the orange block and move it into box"
            instruction = item_list[ni].replace(".pkl", "").split('_')[1]
            # instruction = pkl_data['language_instruction']
            # print(instruction, item_list[ni])
            
            import ipdb; ipdb.set_trace()

            url_ = "http://0.0.0.0:8777/act"
            # url_ = "http://10.140.54.106:10203/act"

            #url_ = "http://10.140.54.77:10205/act"
            # url_ = "http://10.140.54.77:10205/act"
            # url_ = "http://10.140.54.64:10204/act"

            # url_ = "http://10.140.54.6:10203/act"
            # url_ = "http://10.140.54.3:10208/act"
            # url_ = "http://10.140.54.121:10204/act"
            # url_ = "http://10.140.54.121:10204/act"
            # url_ = "http://10.140.54.90:10205/act"
            # url_ = "http://10.140.54.90:10213/act"
            # rgb_img_future = actions_gt[step_cnt+4]["observation"]["exterior_image_1_left"]
            # rgb_img_prev = (data_transform(actions_gt[step_cnt-4]["observation"]["exterior_image_1_left"])*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
            if step_cnt == 0:
                requests.post(url_,json={"full_image": rgb_image_inp1, "instruction": instruction})

            pred_list = []
            avg_nums = 1
            for iii in range(avg_nums):
                action_list = requests.post(
                    # "http://10.140.54.90:10212/act",
                    # "http://10.140.54.6:10203/act", json={"image": rgb_image_resized, "instruction": "pick up the orange block and move it into box"} # Octo
                    # "http://10.140.54.107:10213/act", json={"image": rgb_image_resized, "instruction": instruction}
                    # "http://10.140.54.107:10207/act",json={"image": rgb_image_inp1, "instruction": instruction}
                    # "http://10.140.54.107:10203/act",json={"image": rgb_image_inp1, "instruction": instruction}
                    # "http://10.140.54.107:10202/act",json={"image": rgb_image_inp1, "instruction": instruction}  # finetuned ours lora
                    # "http://10.140.54.6:10204/act",json={"image": rgb_image_inp1, "instruction": instruction}  # finetuned octo-style  nolora
                    url_,json={"full_image": rgb_image_inp1, "instruction": instruction} # finetuned ours nolora
            # http://10.140.54.107:10202
            # http://10.140.54.107:10212 openvla 50k steps
                ).json()    # list

                import ipdb; ipdb.set_trace()
                
                action = np.array(action_list)[0]
                # if len(action.shape) == 2:
                #     action = action[0]
                # print(action, rgb_image_inp.shape)
                # action = np.asarray(action).copy()
                # import ipdb;ipdb.set_trace()
                world_vector = action[:3]

                euler_delta = action[3:6]
                rotation_delta = euler_to_quaternion(action[3:6])
                gripper_closedness_action = 1 - action[-1]

                world_vector_gt, rotation_delta_gt, gripper_closedness_action_gt, pose2 = obtain_gt_delta(actions_gt, step_cnt, None, stride)

                # rotation_delta_gt[0] += 1.0


                euler_delta_gt = quaternion_to_euler_radians(rotation_delta_gt[0], rotation_delta_gt[1], rotation_delta_gt[2], rotation_delta_gt[3]).cpu().numpy()

                # from sklearn import metrics


                s2 = np.concatenate([world_vector, euler_delta, np.asarray([gripper_closedness_action])], axis=0)
                s1 = np.concatenate([world_vector_gt, euler_delta_gt, np.asarray([gripper_closedness_action_gt])], axis=0)
                # print(s1)
                pred_list.append(np.concatenate([world_vector, euler_delta, np.asarray([gripper_closedness_action])], axis=0))
                # import ipdb;ipdb.set_trace()
                s2 = np.stack(pred_list).mean(0)
                if iii == avg_nums - 1:
                    print('mean:', (np.abs(np.stack(pred_list).mean(0) - s1) ).tolist())
                    xxx.append(np.abs(np.stack(pred_list).mean(0)[:6] - s1[:6]).mean())

                print(world_vector, world_vector_gt, np.abs(world_vector - world_vector_gt))
                print(euler_delta, euler_delta_gt, np.abs(euler_delta - euler_delta_gt))
                print(gripper_closedness_action, gripper_closedness_action_gt, np.abs(gripper_closedness_action-gripper_closedness_action_gt))

        print(sum(xxx)/len(xxx))



if __name__ == "__main__":
    # deploy()
    client_example()