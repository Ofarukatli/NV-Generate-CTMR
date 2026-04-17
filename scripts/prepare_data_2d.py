#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def save_nifti(data: np.ndarray, filename: str):
    # For 2D, data is [1, H, W], we want [H, W] transpose for nibabel (X, Y)
    arr = data.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(arr.T, affine)  # transpose so H, W is W, H
    nib.save(nii_img, filename)

def calculate_percentiles(samples, keys, p=[1, 99]):
    print(f"Calculating percentiles {p} for {keys}...")
    values = {k: [] for k in keys}
    
    for sample in tqdm(samples):
        for k in keys:
            data = sample[k]
            # Use random subsample if image is huge, but for 256x256 we can take all
            values[k].extend(data.flatten().tolist())
            
    results = {}
    for k in keys:
        res = np.percentile(values[k], p)
        results[f"{k}_p{p[0]}"] = float(res[0])
        results[f"{k}_p{p[1]}"] = float(res[1])
    return results

def process_pkl(args):
    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pkl, "rb") as f:
        dataset = pickle.load(f)

    # MAISI typically expects a "training" key in the JSON
    # We will put everything in "training" but use "fold" to distinguish
    # fold 0: validation
    # fold 1: training
    # fold 2: test (for inference)
    
    json_data = {"training": []}
    
    split_to_fold = {
        "train": 1,
        "valid": 0,
        "test": 2
    }

    # Step 1: Calculate Global Percentiles
    all_samples = []
    for split in ["train", "valid", "test"]:
        if split in dataset:
            for s in dataset[split]:
                all_samples.append({"image_data": s["image"], "label_data": s["mask"]})
                
    stats = calculate_percentiles(all_samples, ["image_data", "label_data"])
    
    # Step 2: Save NIfTI and Build Main JSON
    json_data = {"training": [], "stats": stats}
    
    for split_name, fold_id in split_to_fold.items():
        if split_name not in dataset:
            continue
            
        samples = dataset[split_name]
        print(f"Processing {split_name} split ({len(samples)} samples)...")
        
        for idx, sample in enumerate(samples):
            target_img = sample["image"]
            cond_img = sample["mask"]
            
            if target_img.shape != cond_img.shape:
                raise ValueError(f"Shape mismatch at {split_name}_{idx}: Image {target_img.shape} != Label {cond_img.shape}")
            
            _, h, w = target_img.shape
            
            img_filename = f"{split_name}_{idx:05d}_target.nii.gz"
            lbl_filename = f"{split_name}_{idx:05d}_cond.nii.gz"
            
            save_nifti(target_img * 1000.0, str(images_dir / img_filename))
            save_nifti(cond_img * 1000.0, str(labels_dir / lbl_filename))
            
            entry = {
                "image": str(Path("images") / img_filename),
                "label": str(Path("labels") / lbl_filename),
                "pseudo_label": str(Path("labels") / lbl_filename),
                "modality": args.modality,
                "spacing": [1.0, 1.0], 
                "dim": [w, h],
                "fold": fold_id,
                "top_region_index": [0.0],
                "bottom_region_index": [1.0]
            }
            json_data["training"].append(entry)

    json_out_path = out_dir / "dataset_maisi.json"
    with open(json_out_path, "w") as f:
        json.dump(json_data, f, indent=4)
        
    print(f"\nSuccess! MAISI dataset written to {out_dir}")
    print(f"Main JSON: {json_out_path}")
    print("Fold mapping: 0=Validation, 1=Training, 2=Test/Inference")

def process_predict_folder(args):
    """Scenario where user just has a folder of T1 images for inference."""
    out_dir = Path(args.out_dir)
    labels_dir = out_dir / "labels" # We treat input T1 as 'label' (condition) in MAISI
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = Path(args.predict_folder)
    extensions = [".png", ".jpg", ".jpeg", ".npy"]
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
    
    if not files:
        print(f"No images found in {input_dir}")
        return

    json_data = {"training": []}
    print(f"Processing {len(files)} files for inference...")

    for idx, fpath in enumerate(files):
        # Load image
        if fpath.suffix.lower() == ".npy":
            arr = np.load(fpath)
        else:
            from PIL import Image
            arr = np.array(Image.open(fpath).convert("L"))
            
        # Normalize if needed (assuming user wants [0, 1000])
        if arr.max() <= 1.0:
            arr = arr * 1000.0
        elif arr.max() > 1000.0:
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 1000.0
            
        lbl_filename = f"predict_{idx:05d}_{fpath.stem}.nii.gz"
        save_nifti(arr.astype(np.float32), str(labels_dir / lbl_filename))
        
        h, w = arr.shape
        entry = {
            "image": str(Path("labels") / lbl_filename), # Dummy, loader needs it
            "label": str(Path("labels") / lbl_filename),
            "pseudo_label": str(Path("labels") / lbl_filename),
            "modality": args.modality,
            "spacing": [1.0, 1.0], 
            "dim": [w, h],
            "fold": 0, # All as val for inference
            "top_region_index": [0.0],
            "bottom_region_index": [1.0]
        }
        json_data["training"].append(entry)

    json_out_path = out_dir / "inference_only.json"
    with open(json_out_path, "w") as f:
        json.dump(json_data, f, indent=4)
    
    print(f"Inference JSON created: {json_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare 2D MRI dataset for MAISI.")
    subparsers = parser.add_subparsers(dest="mode")
    
    # Mode 1: From PKL
    pkl_parser = subparsers.add_parser("from_pkl", help="Convert .pkl to MAISI format")
    pkl_parser.add_argument("--pkl", type=str, required=True)
    pkl_parser.add_argument("--out_dir", type=str, required=True)
    pkl_parser.add_argument("--modality", type=int, default=9)
    
    # Mode 2: From Folder (Inference only)
    folder_parser = subparsers.add_parser("from_folder", help="Create inference JSON from folder of T1 images")
    folder_parser.add_argument("--predict_folder", type=str, required=True)
    folder_parser.add_argument("--out_dir", type=str, required=True)
    folder_parser.add_argument("--modality", type=int, default=9)

    args = parser.parse_args()

    if args.mode == "from_pkl":
        process_pkl(args)
    elif args.mode == "from_folder":
        process_predict_folder(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

