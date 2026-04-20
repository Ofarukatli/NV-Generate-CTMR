#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import monai
from monai.transforms import Compose

from .diff_model_setting import load_config, setup_logging
from .utils import define_instance

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("inference")
    
    args_config = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    
    # Load UNet
    unet = define_instance(args_config, "diffusion_unet_def").to(device)
    if args_config.existing_ckpt_filepath:
        ckpt = torch.load(args_config.existing_ckpt_filepath, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"], strict=True)
        scale_factor = ckpt.get("scale_factor", 1.0)
    else:
        logger.warning("No checkpoint provided, running with random weights.")
        scale_factor = 1.0
        
    unet.eval()
    
    # Load input list and stats
    with open(args_config.json_data_list) as f:
        meta = json.load(f)
        data_list = meta["training"]
        stats = meta.get("stats", {})
        
    t2_p1, t2_p99 = stats.get("image_data_p1", 0.0), stats.get("image_data_p99", 1000.0)
    t1_p1, t1_p99 = stats.get("label_data_p1", 0.0), stats.get("label_data_p99", 1000.0)
        
    out_dir = os.path.join(args_config.output_dir, args_config.output_prefix)
    os.makedirs(out_dir, exist_ok=True)
    
    num_steps = args_config.diffusion_unet_inference.get("num_inference_steps", 50)
    dt = 1.0 / num_steps
    
    logger.info(f"Starting Euler ODE Inference ({num_steps} steps) on {len(data_list)} subjects...")
    
    transforms = Compose([
        monai.transforms.LoadImaged(keys=["label"]),
        monai.transforms.EnsureChannelFirstd(keys=["label"]),
    ])
    
    for idx, item in enumerate(data_list):
        if item.get("fold") != 2: # Only process test fold
            continue
            
        cond_path = os.path.join(args_config.data_base_dir, item["label"])
        if not os.path.exists(cond_path):
            continue
            
        data = transforms({"label": cond_path})
        cond_tensor = data["label"].unsqueeze(0).to(device) # [1, 1, 256, 256]
        
        # Percentile Normalization (to 0-1)
        cond_tensor = (cond_tensor - t1_p1) / (t1_p99 - t1_p1 + 1e-8)
        cond_tensor = torch.clamp(cond_tensor, 0, 1) * scale_factor
        
        # Start from Noise (t=0)
        x_t = torch.randn_like(cond_tensor).to(device)
        
        modality = torch.full((1,), item.get("modality", 9), dtype=torch.long).to(device)
        
        with torch.no_grad():
            for i in range(num_steps):
                t_val = i / num_steps
                t_tensor = torch.ones((1,), device=device) * t_val * 1000.0
                
                # Concatenate the clean condition onto the noisy target
                unet_x = torch.cat([x_t, cond_tensor], dim=1)
                
                # Model predicts velocity v
                v = unet(unet_x, timesteps=t_tensor, class_labels=modality)
                
                # Euler step: x_{t+dt} = x_t + v * dt
                x_t = x_t + v * dt
                    
        # De-normalize output from [0, 1] back to T2 range
        out_array = x_t.squeeze().cpu().numpy() / scale_factor
        out_array = out_array * (t2_p99 - t2_p1) + t2_p1
        
        # Use original subject filename for the output
        orig_filename = Path(item["label"]).name.replace(".nii.gz", "")
        out_name = os.path.join(out_dir, f"{orig_filename}_inferred.nii.gz")
        
        # Transpose back for NIfTI
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
