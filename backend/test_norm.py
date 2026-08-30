import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import xarray as xr
import json

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

def norm_channel(arr, mask):
    valid = arr[~mask]
    if len(valid) == 0: return arr
    mean, std = np.mean(valid), np.std(valid)
    if std == 0: std = 1
    res = arr.copy()
    res[~mask] = (res[~mask] - mean) / std
    res[mask] = 0
    return res

def main():
    device = torch.device("cpu")
    model = ResNet50_UNet_Pro(in_channels=5, out_channels=15).to(device)
    model.load_state_dict(torch.load("BEST_ocean_model_1YR (1).pth", map_location=device, weights_only=True))
    model.eval()

    target_date = "2026-08-30"
    ds_thetao = xr.open_dataset(f"temp_thetao_{target_date}.nc")
    ds_so = xr.open_dataset(f"temp_so_{target_date}.nc")
    ds_cur = xr.open_dataset(f"temp_cur_{target_date}.nc")
    ds_zos = xr.open_dataset(f"temp_zos_{target_date}.nc")
    
    sst = ds_thetao['thetao'].isel(time=0, depth=0).values
    mask = np.isnan(sst)
    
    sst_n = norm_channel(sst, mask)
    sss_n = norm_channel(ds_so['so'].isel(time=0, depth=0).values, mask)
    ssh_n = norm_channel(ds_zos['zos'].isel(time=0).values, mask)
    u_n = norm_channel(ds_cur['uo'].isel(time=0, depth=0).values, mask)
    v_n = norm_channel(ds_cur['vo'].isel(time=0, depth=0).values, mask)
    
    input_array = np.stack([sst_n, sss_n, ssh_n, u_n, v_n], axis=0)
    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_norm = model(input_tensor)
        
    pred_norm_np = prediction_norm.squeeze(0).cpu().numpy()
    
    print("\n--- NORMALIZED PREDICTION STATS ---")
    for i in range(15):
        layer = pred_norm_np[i].copy()
        layer[mask] = np.nan
        valid = layer[~np.isnan(layer)]
        print(f"Layer {i}: mean={np.mean(valid):.4f}, std={np.std(valid):.4f}")

if __name__ == "__main__":
    main()
