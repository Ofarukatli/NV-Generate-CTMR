#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time

import nibabel as nib
import numpy as np
import torch
import monai
from monai.transforms import Compose
from monai.networks.schedulers import RFlowScheduler, DDPMScheduler

from diff_model_setting import load_config, setup_logging
from utils import define_instance

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("inference")
    
    args_config = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    
    # Load UNet
    unet = define_instance(args_config, "diffusion_unet_def").to(device)
    if args_config.existing_ckpt_filepath:
        ckpt = torch.load(args_config.existing_ckpt_filepath, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
        scale_factor = ckpt.get("scale_factor", 1.0)
    else:
        logger.warning("No checkpoint provided, running with random weights.")
        scale_factor = 1.0
        
    unet.eval()
    noise_scheduler = define_instance(args_config, "noise_scheduler")
    
    # Load input list
    with open(args_config.json_data_list) as f:
        data_list = json.load(f)["training"]
        
    out_dir = os.path.join(args_config.output_dir, args_config.output_prefix)
    os.makedirs(out_dir, exist_ok=True)
    
    logger.info(f"Starting inference on {len(data_list)} subjects...")
    
    transforms = Compose([
        monai.transforms.LoadImaged(keys=["label"]),
        monai.transforms.EnsureChannelFirstd(keys=["label"]),
        monai.transforms.ScaleIntensityRangePercentsd(keys=["label"], lower=0, upper=100, b_min=0.0, b_max=1.0, clip=True)
    ])
    
    for idx, item in enumerate(data_list):
        cond_path = os.path.join(args_config.data_base_dir, item["label"])
        if not os.path.exists(cond_path):
            continue
            
        data = transforms({"label": cond_path})
        cond_tensor = data["label"].unsqueeze(0).to(device) # [1, 1, 256, 256]
        cond_tensor = cond_tensor * scale_factor
        
        # Noise
        noise = torch.randn_like(cond_tensor).to(device)
        
        spacing = torch.FloatTensor(item.get("spacing", [1.0, 1.0])).unsqueeze(0).to(device) * 1e2
        if spacing.shape[1] == 2:
            spacing = torch.cat([spacing, torch.zeros((1, 1), device=device)], dim=1)

        top_idx = torch.FloatTensor([item.get("top_region_index", [0.0])[0]]).unsqueeze(0).to(device) * 1e2
        bot_idx = torch.FloatTensor([item.get("bottom_region_index", [1.0])[0]]).unsqueeze(0).to(device) * 1e2
        if top_idx.shape[1] == 1:
            top_idx = torch.cat([top_idx, torch.zeros((1, 3), device=device)], dim=1)
            bot_idx = torch.cat([bot_idx, torch.zeros((1, 3), device=device)], dim=1)
        modality = torch.full((1,), item.get("modality", 9), dtype=torch.long).to(device)
        
        unet_inputs = {
            "spacing_tensor": spacing,
            "class_labels": modality,
            "top_region_index_tensor": top_idx,
            "bottom_region_index_tensor": bot_idx
        }
        
        num_inference_steps = args_config.diffusion_unet_inference.get("num_inference_steps", 30)
        noise_scheduler.set_timesteps(num_inference_steps)
        
        current_img = noise
        with torch.no_grad():
            for t in noise_scheduler.timesteps:
                # Concatenate the clean condition onto the noisy target
                unet_x = torch.cat([current_img, cond_tensor], dim=1)
                
                model_output = unet(unet_x, timesteps=torch.Tensor((t,)).to(device), **unet_inputs)
                
                if isinstance(noise_scheduler, RFlowScheduler):
                    current_img, _ = noise_scheduler.step(model_output, t, current_img)
                else:
                    current_img = noise_scheduler.step(model_output, t, current_img).prev_sample
                    
        # Save output
        out_array = current_img.squeeze().cpu().numpy() / scale_factor
        out_name = os.path.join(out_dir, f"inferred_{idx:04d}.nii.gz")
        
        nii_img = nib.Nifti1Image(out_array.T, np.eye(4))
        nib.save(nii_img, out_name)
        logger.info(f"Saved {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_config_path", type=str, required=True)
    parser.add_argument("-c", "--model_config_path", type=str, required=True)
    parser.add_argument("-t", "--model_def_path", type=str, required=True)
    args = parser.parse_args()
    
    infer(args)
