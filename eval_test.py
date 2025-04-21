import os
import torch
import json, functools
import numpy as np
from PIL import Image
from robovlms.train.base_trainer import BaseTrainer
from robovlms.data.data_utils import preprocess_image
from robovlms.data.data_utils import get_text_function

np.set_printoptions(precision=3, suppress=True)
act_q01 = np.array([-0.08000017702579498,
            -0.20688602328300476,
            -0.16130231320858002,
            -0.5491514801979065,
            -0.2616570293903351,
            -0.44785112142562866,
            0.0])
act_q99 = np.array([0.12800641357898712,
            0.17082400619983673,
            0.1557823121547699,
            0.3282812535762787,
            0.2638643980026245,
            0.5106926560401917,
            1.0])

configs = json.load(open('configs/kosmos_ph_post_train_lab.json', 'r'))
pretrained_path = '/mnt/afs/share_data/tongronglei/work/RoboVLMs/runs/checkpoints/oxe_post_train/kosmos/kosmos/lab_sft/2025-04-14/15-13/epoch=28-step=100000.ckpt'

if os.path.isdir(pretrained_path):
    target_ckpt_path = pretrained_path.replace(".ckpt", ".pt")
    from robovlms.utils.zero_to_fp32 import (
        convert_zero_checkpoint_to_fp32_state_dict,
    )

    print(f"converting {pretrained_path} to {target_ckpt_path}")
    convert_zero_checkpoint_to_fp32_state_dict(pretrained_path, target_ckpt_path)
    pretrained_path = target_ckpt_path

configs['model_load_path'] = pretrained_path

model = BaseTrainer.from_checkpoint(pretrained_path, configs.get("model_load_source", "torch"), configs)
model.to("cuda")
model.eval()

image_fn = functools.partial(
    preprocess_image,
    image_processor=model.model.image_processor,
    model_type=configs["model"],
)
text_fn = get_text_function(model.model.tokenizer, configs["model"])

pkl_path = "/mnt/afs/share_data/xuyuan2/nips_pkl/20250409170732_pick up the carrot into the the box.pkl"
import pickle
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.CenterCrop(480),
    transforms.Resize((224, 224)),
])
data_pkl = pickle.load(open(pkl_path, 'rb'))
frames = []
wrists = []
for step in data_pkl["steps"]:
    img = step["observation"]["exterior_image_1_left"]
    img = Image.fromarray(img)
    img = transform(img)
    frames.append(img)
    wrist = step["observation"]["exterior_image_1_wrist"]
    wrist = Image.fromarray(wrist)
    wrist = transform(wrist)
    wrists.append(wrist)

prompt = "pick up the carrot into the the box.pkl"
prompt = f"In: What action should the robot take to {prompt.lower()}?\nOut:"
text_tensor, attention_mask = text_fn([prompt])

# from collections import deque
# images = deque(maxlen=configs['window_size'])

for image, wrist in zip(frames, wrists):
    input_dict = dict()

    # images.append(image)
    # image_tensors = image_fn(images).unsqueeze(0)
    image_tensors = image_fn([image]).unsqueeze(0)

    input_dict["rgb"] = image_tensors
    input_dict["text"] = text_tensor
    input_dict['text_mask'] = attention_mask

    ## if wrist camera is available
    # wrist_image: Image.Image = Image.fromarray((torch.rand(224, 224, 3) * 255).numpy().astype('uint8'))
    wrist_tensors = image_fn([wrist]).unsqueeze(0)
    input_dict["hand_rgb"] = wrist_tensors

    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.cuda()

    with torch.no_grad():
        action_chunk = model.inference_step(input_dict)["action"]

    # import ipdb;ipdb.set_trace()

    action_chunk = torch.cat([action_chunk[0], 2*(torch.nn.functional.sigmoid(action_chunk[1])>0.5).float()-1], dim=-1)
    action = action_chunk.select(dim=-3, index=0).select(dim=-2, index=0).squeeze().cpu().numpy()

    if configs['train_dataset']['norm']:
        action = 0.5 * (action + 1) * (act_q99 - act_q01) + act_q01
 
    print(action)


