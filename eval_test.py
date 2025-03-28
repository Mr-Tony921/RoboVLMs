import os
import torch
import json, functools
import numpy as np
from PIL import Image
from robovlms.train.base_trainer import BaseTrainer
from robovlms.data.data_utils import preprocess_image
from robovlms.data.data_utils import get_text_function

np.set_printoptions(precision=3)
act_q01 = np.array([-0.086, -0.249, -0.178, -0.539, -0.283, -0.468, 0.0])
act_q99 = np.array([0.137, 0.175, 0.164, 0.332, 0.276, 0.553, 1.0])

configs = json.load(open('configs/kosmos_ph_post_train_lab.json', 'r'))
pretrained_path = 'runs/checkpoints/oxe_post_train/kosmos/kosmos/lab_sft/2025-03-27/20-14/epoch=684-step=50000.pt'

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
prompt = "pickup the bottle on the table"
prompt = f"In: What action should the robot take to {prompt.lower()}?\nOut:"
text_tensor, attention_mask = text_fn([prompt])

for step in range(200):
    input_dict = dict()
    
    image: Image.Image = Image.fromarray((torch.rand(224, 224, 3) * 255).numpy().astype('uint8'))
    image = image_fn([image]).unsqueeze(0)
    
    input_dict["rgb"] = image
    input_dict["text"] = text_tensor
    input_dict['text_mask'] = attention_mask

    ### if wrist camera is available
    # wrist_image: Image.Image = Image.fromarray((torch.rand(224, 224, 3) * 255).numpy().astype('uint8'))
    # wrist_image = image_fn([wrist_image]).unsqueeze(0)
    # input_dict["hand_rgb"] = wrist_image

    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.cuda()

    with torch.no_grad():
        action_chunk = model.inference_step(input_dict)["action"]

    action_chunk = torch.cat([action_chunk[0], 2*(action_chunk[1]>0.2).float()-1], dim=-1)
    action = action_chunk.select(dim=-2, index=0).squeeze().cpu().numpy()

    if configs['train_dataset']['norm_action']:
        action = 0.5 * (action + 1) * (act_q99 - act_q01) + act_q01
 
    print(action)


