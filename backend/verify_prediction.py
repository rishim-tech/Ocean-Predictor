import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import xarray as xr
import json

# MODEL ARCHITECTURE
class DecoderBlockPro(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        )
        self.conv = nn.Sequential(
            nn.Conv2d((in_channels // 2) + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet50_UNet_Pro(nn.Module):
    def __init__(self, in_channels=5, out_channels=15):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = resnet.conv1.weight
            self.conv1.weight[:, 3:] = resnet.conv1.weight[:, :2] 
        self.bn1 = resnet.bn1; self.relu = resnet.relu; self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1; self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3; self.layer4 = resnet.layer4
        self.dec4 = DecoderBlockPro(2048, 1024, 512); self.dec3 = DecoderBlockPro(512, 512, 256)
        self.dec2 = DecoderBlockPro(256, 256, 128); self.dec1 = DecoderBlockPro(128, 64, 64)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        original_size = (x.shape[2], x.shape[3])
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        skip1 = self.layer1(x1); skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2); bottleneck = self.layer4(skip3)
        d4 = self.dec4(bottleneck, skip3); d3 = self.dec3(d4, skip2)
        d2 = self.dec2(d3, skip1); d1 = self.dec1(d2, x0)
        out = self.final_up(d1)
        if out.shape[2:] != original_size: out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=True)
        return self.final_conv(out)


def get_stats(arr):
    arr_finite = arr[np.isfinite(arr)]
    if len(arr_finite) == 0:
        return {}
    return {
        "min": float(np.min(arr_finite)),
        "max": float(np.max(arr_finite)),
        "mean": float(np.mean(arr_finite)),
        "median": float(np.median(arr_finite)),
        "std": float(np.std(arr_finite)),
        "p1": float(np.percentile(arr_finite, 1)),
        "p5": float(np.percentile(arr_finite, 5)),
        "p50": float(np.percentile(arr_finite, 50)),
        "p95": float(np.percentile(arr_finite, 95)),
        "p99": float(np.percentile(arr_finite, 99)),
        "nan_count": int(np.sum(np.isnan(arr))),
        "inf_count": int(np.sum(np.isinf(arr))),
        "finite_count": int(len(arr_finite))
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load model
    model = ResNet50_UNet_Pro(in_channels=5, out_channels=15).to(device)
    model.load_state_dict(torch.load("BEST_ocean_model_1YR (1).pth", map_location=device, weights_only=True))
    model.eval()

    # 2. Load stats
    with open("depth_stats.json", "r") as f:
        stats = json.load(f)
    means = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 15, 1, 1).to(device)
    stds = torch.tensor(stats["std"], dtype=torch.float32).view(1, 15, 1, 1).to(device)
    
    print("depth_stats means:", stats["mean"])
    print("depth_stats stds:", stats["std"])

    # 3. Load Inputs
    target_date = "2026-08-30"
    file_thetao = f"temp_thetao_{target_date}.nc"
    file_so = f"temp_so_{target_date}.nc"
    file_cur = f"temp_cur_{target_date}.nc"
    file_zos = f"temp_zos_{target_date}.nc"

    ds_thetao = xr.open_dataset(file_thetao)
    ds_so = xr.open_dataset(file_so)
    ds_cur = xr.open_dataset(file_cur)
    ds_zos = xr.open_dataset(file_zos)
    
    sst_raw = ds_thetao['thetao'].isel(time=0, depth=0).values
    print("SST RAW Stats:", get_stats(sst_raw))
    print("SST NaN count before fill:", np.isnan(sst_raw).sum())
    
    sst = ds_thetao['thetao'].isel(time=0, depth=0).fillna(0).values
    sss = ds_so['so'].isel(time=0, depth=0).fillna(0).values
    u_cur = ds_cur['uo'].isel(time=0, depth=0).fillna(0).values
    v_cur = ds_cur['vo'].isel(time=0, depth=0).fillna(0).values
    ssh = ds_zos['zos'].isel(time=0).fillna(0).values
    
    print("Input channels order: SST, SSS, SSH, U, V")
    print("SST input stats (filled with 0):", get_stats(sst))
    print("SSS input stats:", get_stats(sss))
    print("SSH input stats:", get_stats(ssh))
    
    input_array = np.stack([sst, sss, ssh, u_cur, v_cur], axis=0)
    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        prediction_norm = model(input_tensor)
        prediction_denorm = (prediction_norm * stds) + means
        
    pred_norm_np = prediction_norm.squeeze(0).cpu().numpy()
    pred_denorm_np = prediction_denorm.squeeze(0).cpu().numpy()
    
    print("\n--- PREDICTION LAYER STATS (DENORMALIZED) ---")
    
    # 5. Apply land mask
    land_mask = (sst == 0.0)
    
    for i in range(15):
        layer_denorm = pred_denorm_np[i].copy()
        layer_denorm[land_mask] = np.nan
        
        layer_norm = pred_norm_np[i].copy()
        layer_norm[land_mask] = np.nan
        
        print(f"\nLayer {i} (depth ~{stats['depths'][i]} m)")
        print(f"  Denormalized Stats: {get_stats(layer_denorm)}")
        print(f"  Normalized Stats: {get_stats(layer_norm)}")
        print(f"  Expected mean from json: {stats['mean'][i]}, std: {stats['std'][i]}")

if __name__ == "__main__":
    main()
