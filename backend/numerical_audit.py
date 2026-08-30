import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import torch
import copernicusmarine
import xarray as xr
import sys
from main import ResNet50_UNet_Pro

def run_audit():
    # 1. Load Depth Stats
    STATS_PATH = "depth_stats.json"
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
    
    means = stats["mean"]
    stds = stats["std"]
    depths = stats["depths"]
    
    # 2. Load Model
    device = torch.device("cpu")
    model = ResNet50_UNet_Pro(in_channels=5, out_channels=16).to(device)
    model.load_state_dict(torch.load("ocean_model_16_channel_stage2.pth", map_location=device, weights_only=True))
    model.eval()
    
    # 3. Predict on mock/cached 2026-08-30 data
    ds_thetao = xr.open_dataset("temp_thetao_2026-08-30.nc")
    ds_so = xr.open_dataset("temp_so_2026-08-30.nc")
    ds_cur = xr.open_dataset("temp_cur_2026-08-30.nc")
    ds_zos = xr.open_dataset("temp_zos_2026-08-30.nc")
    
    sst_raw = ds_thetao['thetao'].isel(time=0, depth=0)
    sss_raw = ds_so['so'].isel(time=0, depth=0)
    u_raw = ds_cur['uo'].isel(time=0, depth=0)
    v_raw = ds_cur['vo'].isel(time=0, depth=0)
    ssh_raw = ds_zos['zos'].isel(time=0)
    
    mask = np.isnan(sst_raw.values)
    
    sst = sst_raw.fillna(0).values
    sss = sss_raw.fillna(0).values
    u_cur = u_raw.fillna(0).values
    v_cur = v_raw.fillna(0).values
    ssh = ssh_raw.fillna(0).values
    
    input_array = np.stack([sst, sss, ssh, u_cur, v_cur], axis=0)
    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor).squeeze(0).cpu().numpy()
        
    # Denormalize ONLY 0, 14, 15
    for c in range(16):
        if c in [0, 14, 15]:
            prediction[c] = (prediction[c] * stds[c]) + means[c]
        prediction[c][mask] = np.nan
        
    print(f"{'Depth':>8} | {'Ch':>3} | {'Pred Min':>10} | {'Pred Max':>10} | {'Mean':>10} | {'Std':>10} | {'Ref Mean':>10} | {'MAE':>10} | {'RMSE':>10} | {'Color Min':>10} | {'Color Max':>10} | {'Status':>15}")
    print("-" * 145)
    
    failed_layers = []
    
    for c in range(16):
        layer = prediction[c]
        valid = layer[~np.isnan(layer)]
        
        p_min = np.min(valid)
        p_max = np.max(valid)
        p_mean = np.mean(valid)
        p_std = np.std(valid)
        
        # Colorbar limits from frontend Auto-Contrast (2nd to 98th percentile)
        c_min = np.percentile(valid, 2)
        c_max = np.percentile(valid, 98)
        
        # Reference Data (we can't fetch it due to copernicusmarine bug)
        ref_mean = "N/A"
        mae = "N/A"
        rmse = "N/A"
        status = "NOT VALIDATED" # As instructed if ref data is unavailable
        
        # However, we check if prediction stats are somewhat close to target training stats
        target_mean = means[c]
        target_std = stds[c]
        
        if abs(p_mean - target_mean) > (2.0 * target_std):
            failed_layers.append((depths[c], c, "Mean divergence > 2 std"))
            status = "FAILED"
        elif np.isnan(p_mean):
            failed_layers.append((depths[c], c, "NaN output"))
            status = "FAILED"
        
        print(f"{depths[c]:8.3f} | {c:3} | {p_min:10.3f} | {p_max:10.3f} | {p_mean:10.3f} | {p_std:10.3f} | {ref_mean:>10} | {mae:>10} | {rmse:>10} | {c_min:10.3f} | {c_max:10.3f} | {status:>15}")

    print("\n--- FAILED LAYERS ---")
    if not failed_layers:
        print("None. All layers passed baseline statistical sanity checks.")
    else:
        for f in failed_layers:
            print(f"Layer L{f[1]} ({f[0]}m): {f[2]}")

if __name__ == "__main__":
    run_audit()
