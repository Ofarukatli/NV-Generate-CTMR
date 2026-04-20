import torch
import os

ckpt_path = '/auto/data2/ademirtas/oatli/NV-Generate-CTMR-nol1/NV-Generate-CTMR/models/diff_unet_2d_t1_t2.pt'

if not os.path.exists(ckpt_path):
    print(f"Error: File not found at {ckpt_path}")
else:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    weights = ckpt['unet_state_dict']
    
    # Calculate average absolute mean of weights
    means = [v.abs().mean().item() for v in weights.values() if hasattr(v, 'abs')]
    avg_mean = sum(means) / len(means)
    
    print("-" * 30)
    print(f"Checkpoint Info:")
    print(f"  Epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  Last Logged Loss: {ckpt.get('loss', 'unknown')}")
    print(f"  Mean Weight Amplitude: {avg_mean:.8f}")
    print("-" * 30)
    
    if avg_mean < 1e-4:
        print("STATUS: COLLAPSED (Weights are near-zero. Do NOT resume.)")
    elif torch.isnan(torch.tensor(means)).any():
        print("STATUS: CORRUPTED (NaNs found. Do NOT resume.)")
    else:
        print("STATUS: OK (You can safely resume training.)")
    print("-" * 30)
