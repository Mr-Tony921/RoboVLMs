"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

import functools
from robovlms.train.base_trainer import BaseTrainer
from robovlms.data.data_utils import preprocess_image
from robovlms.data.data_utils import get_text_function
import torchvision.transforms as transforms

def get_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class RoboVLMServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.configs = json.load(open(cfg.model_configs, 'r'))
        # Load model
        self.model = BaseTrainer.from_checkpoint(cfg.pretrained_checkpoint, self.configs.get("model_load_source", "torch"), self.configs)
        self.model.to("cuda")
        self.model.eval()

        # image_fn
        self.image_fn = functools.partial(
            preprocess_image,
            image_processor=self.model.model.image_processor,
            model_type=self.configs["model"],
        )
        self.transform = transforms.Compose([
            transforms.CenterCrop(480),
            transforms.Resize((224, 224)),
        ])

        # text_fn
        self.text_fn = get_text_function(self.model.model.tokenizer, self.configs["model"])
        
        # use wrist
        self.use_hand_rgb = self.configs.get("use_hand_rgb", False)

        # norm
        self.act_q01 = cfg.act_q01
        self.act_q99 = cfg.act_q99


    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            input_dict = dict()
            observation = payload
            instruction = observation["instruction"]
            text_tensor, attention_mask = self.text_fn([get_prompt(instruction)])
            input_dict["text"] = text_tensor
            input_dict['text_mask'] = attention_mask

            image = observation["image"]
            image = Image.fromarray(image)
            image = self.transform(image)
            image_tensors = self.image_fn([image]).unsqueeze(0)
            input_dict["rgb"] = image_tensors

            if self.use_hand_rgb and "wrist" in observation:
                wrist = observation["wrist"]
                wrist = Image.fromarray(wrist)
                wrist = self.transform(wrist)
                wrist_tensors = self.image_fn([wrist]).unsqueeze(0)
                input_dict["hand_rgb"] = wrist_tensors

            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.cuda()

            with torch.no_grad():
                action_chunk = self.model.inference_step(input_dict)["action"]

            action_chunk = torch.cat([action_chunk[0], 2*(torch.nn.functional.sigmoid(action_chunk[1])>0.5).float()-1], dim=-1)
            action = action_chunk.select(dim=-3, index=0).select(dim=-2, index=0).squeeze().cpu().numpy()

            if self.configs['train_dataset']['norm']:
                action = 0.5 * (action + 1) * (self.act_q99 - self.act_q01) + self.act_q01

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8777                                                    # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "robovlm"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    model_configs: Union[str, Path] = "configs/kosmos_ph_post_train_lab.json" # Model configs path

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

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = RoboVLMServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
