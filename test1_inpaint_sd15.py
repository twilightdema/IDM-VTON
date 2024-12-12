# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from ip_adapter.ip_adapter import Resampler

import argparse
import logging
import os
import torch.utils.data as data
import torchvision
import json
import accelerate
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
import diffusers
from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer

from diffusers.utils.import_utils import is_xformers_available

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline



logger = get_logger(__name__, log_level="INFO")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEBUG = True

def _debug_print(s):
    if DEBUG:
        print(s)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default= "result_sd15/checkpoint-best-train-loss",required=False,)
    parser.add_argument("--width",type=int,default=384,)
    parser.add_argument("--height",type=int,default=512,)
    parser.add_argument("--num_inference_steps",type=int,default=30,)
    parser.add_argument("--output_dir",type=str,default="inference_result_sd15",)
    parser.add_argument("--unpaired",action="store_true",)
    parser.add_argument("--data_dir",type=str,default="/home/omnious/workspace/yisol/Dataset/zalando")
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--test_batch_size", type=int, default=2,)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    args = parser.parse_args()


    return args

def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


class VitonHDTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super(VitonHDTestDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()

        with open(
            os.path.join(dataroot_path, phase, "vitonhd_" + phase + "_tagged.json"), "r"
        ) as file1:
            data1 = json.load(file1)

        annotation_list = [
            "sleeveLength",
            "neckLine",
            "item",
        ]

        self.annotation_pair = {}
        for k, v in data1.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                self.annotation_pair[elem["file_name"]] = annotation_str

        self.order = order
        self.toTensor = transforms.ToTensor()

        im_names = []
        c_names = []
        dataroot_names = []


        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.clip_processor = CLIPImageProcessor()
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        if c_name in self.annotation_pair:
            cloth_annotation = self.annotation_pair[c_name]
        else:
            cloth_annotation = "shirts"
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name))

        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))
        image = self.transform(im_pil_big)

        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-mask", im_name.replace('.jpg','_mask.png'))).resize((self.width,self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]
        mask = 1-mask
        im_mask = image * mask
 
        pose_img = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", im_name)
        )
        # TODO(chulayuth) pose_img should be resize here instead in graph processing step?
        pose_img = self.transform(pose_img)  # [-1,1]
 
        result = {}

        result["c_name"] = c_name
        result["im_name"] = im_name
        result["image"] = image
        result["cloth_pure"] = self.transform(cloth)
        result["cloth"] = self.clip_processor(images=cloth, return_tensors="pt").pixel_values
        result["inpaint_mask"] =1-mask
        result["im_mask"] = im_mask
        result["caption_cloth"] = "a photo of " + cloth_annotation
        result["caption"] = "model is wearing a " + cloth_annotation
        result["pose_img"] = pose_img


        return result

    def __len__(self):
        # model images + cloth image
        return len(self.im_names)




def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    #     args.mixed_precision = accelerator.mixed_precision
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    #     args.mixed_precision = accelerator.mixed_precision
    
    test_dataset = VitonHDTestDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order="unpaired" if args.unpaired else "paired",
        size=(args.height, args.width),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
    )

    _debug_print(f'>>>>>>>>>>>>> accelerator.device = {accelerator.device}')

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        # "sd-legacy/stable-diffusion-inpainting",
        # revision="fp16",
        torch_dtype=torch.float16,
    ).to(accelerator.device)


    RUN_COUNT = 10

    count = 0

    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for sample in test_dataloader:

                    count = count + 1
                    if count > RUN_COUNT:
                        break

                    # Also save the input images.
                    batch_size = sample["image"].shape[0]
                    _debug_print(f'cloth.shape = {sample["cloth"].shape}')
                    _debug_print(f'inpaint_mask.shape = {sample["inpaint_mask"].shape}')
                    for b in range(batch_size):
                      image = sample['image'][b]
                      cloth = sample['cloth'][b][0]
                      inpaint_mask = sample['inpaint_mask'][b]
                      torchvision.utils.save_image(image, os.path.join(args.output_dir, f'test_{count}_input_image_{b}.jpg'))
                      torchvision.utils.save_image(cloth, os.path.join(args.output_dir, f'test_{count}_input_cloth_{b}.jpg'))
                      torchvision.utils.save_image(inpaint_mask, os.path.join(args.output_dir, f'test_{count}_input_inpaint_mask_{b}.jpg'))

                    img_list = []
                    inp_list = []
                    for i in range(sample['image'].shape[0]):
                      image = sample['image'][i]
                      inpaint_mask = sample['inpaint_mask'][i]
                      img_list.append(image)
                      inp_list.append(inpaint_mask)
                    
                    prompt = sample["caption"]

                    #image and mask_image should be PIL images.
                    #The mask structure is white for inpainting and black for keeping as is
                    images = pipe(prompt=prompt, image=img_list[0], mask_image=inp_list[0]).images


                    for i in range(len(images)):
                        x_sample = pil_to_tensor(images[i])
                        torchvision.utils.save_image(x_sample, os.path.join(args.output_dir, f'test_{count}_output_{i}.jpg'))
                


if __name__ == "__main__":
    main()
