import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import xarray as xr
import sys
from verify_prediction import ResNet50_UNet_Pro

def check():
    device = torch.device("cpu")
    model = ResNet50_UNet_Pro(in_channels=5, out_channels=15).to(device)
    model.load_state_dict(torch.load("BEST_ocean_model_1YR (1).pth", map_location=device, weights_only=True))
    model.eval()
    
    ds_thetao = xr.open_dataset("temp_thetao_2026-08-30.nc")
    ds_so = xr.open_dataset("temp_so_2026-08-30.nc")
    ds_cur = xr.open_dataset("temp_cur_2026-08-30.nc")
    ds_zos = xr.open_dataset("temp_zos_2026-08-30.nc")
    
    sst = ds_thetao['thetao'].isel(time=0, depth=0).fillna(0).values
    sss = ds_so['so'].isel(time=0, depth=0).fillna(0).values
    u_cur = ds_cur['uo'].isel(time=0, depth=0).fillna(0).values
    v_cur = ds_cur['vo'].isel(time=0, depth=0).fillna(0).values
    ssh = ds_zos['zos'].isel(time=0).fillna(0).values
    
    mask = (sst == 0)
    
    input_array = np.stack([sst, sss, ssh, u_cur, v_cur], axis=0)
    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor).squeeze(0).cpu().numpy()
        
    print("--- RAW MODEL PREDICTION STATS (NO DENORMALIZATION) ---")
    for c in range(15):
        layer = prediction[c]
        valid = layer[~mask]
        if len(valid) == 0:
            continue
        p_mean = np.mean(valid)
        p_std = np.std(valid)
        print(f"Layer {c}: Mean = {p_mean:.3f} °C, Std = {p_std:.3f} °C")

if __name__ == "__main__":
    check()
