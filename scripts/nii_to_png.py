import os
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def convert_dir(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = list(input_path.glob("*.nii.gz"))
    print(f"Converting {len(files)} NIfTI files to PNG...")
    
    for f in tqdm(files):
        try:
            # Load NIfTI
            img = nib.load(str(f))
            data = img.get_fdata()
            
            # For 2D slices, data is often [W, H] after our transpose
            # We want to handle any orientation
            if data.ndim == 3:
                # If there's a dummy 3rd dimension [W, H, 1]
                data = np.squeeze(data)
            
            # Transpose back if necessary (to H, W for PIL)
            data = data.T
            
            # Normalize to 0-255 for viewing
            # We use global min/max of the slice
            d_min, d_max = data.min(), data.max()
            if d_max > d_min:
                data_norm = (data - d_min) / (d_max - d_min) * 255.0
            else:
                data_norm = data * 0.0
                
            img_out = Image.fromarray(data_norm.astype(np.uint8))
            
            # Save as PNG
            out_name = f.name.replace(".nii.gz", ".png")
            img_out.save(output_path / out_name)
        except Exception as e:
            print(f"Error converting {f.name}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./outputs/unet_2d")
    parser.add_argument("--output_dir", type=str, default="./outputs/unet_2d_png")
    args = parser.parse_args()
    
    convert_dir(args.input_dir, args.output_dir)
