import os
import time
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import psutil

# Ensure OpenMP handles multiple loads gracefully (required for some Windows setups)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from verify_prediction import ResNet50_UNet_Pro

class GLORYS_1000m_Dataset(Dataset):
    def __init__(self, nc_file):
        print(f"Loading native GLORYS dataset: {nc_file}")
        self.ds_theta = xr.open_dataset(nc_file)
        self.ds_so = xr.open_dataset(nc_file.replace('thetao', 'so'))
        self.ds_cur = xr.open_dataset(nc_file.replace('thetao', 'cur'))
        self.ds_zos = xr.open_dataset(nc_file.replace('thetao', 'zos'))
        
        self.in_channels = 5
        self.out_channels = 16
        
        depths = self.ds_theta.depth.values
        target_depths_meters = [0.5, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 925.0]
        self.target_indices = []
        for d in target_depths_meters:
            self.target_indices.append((np.abs(depths - d)).argmin())
            
        self.idx_902 = (np.abs(depths - 902.339)).argmin()
        self.idx_1062 = (np.abs(depths - 1062.44)).argmin()
        self.w_902 = (1062.44 - 1000.0) / (1062.44 - 902.339)
        self.w_1062 = (1000.0 - 902.339) / (1062.44 - 902.339)

    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        sst_raw = self.ds_theta['thetao'].isel(time=idx, depth=0)
        sss_raw = self.ds_so['so'].isel(time=idx, depth=0)
        u_raw = self.ds_cur['uo'].isel(time=idx, depth=0)
        v_raw = self.ds_cur['vo'].isel(time=idx, depth=0)
        ssh_raw = self.ds_zos['zos'].isel(time=idx)
        
        sst = sst_raw.fillna(0).values
        sss = sss_raw.fillna(0).values
        u = u_raw.fillna(0).values
        v = v_raw.fillna(0).values
        ssh = ssh_raw.fillna(0).values
        
        input_tensor = torch.tensor(np.stack([sst, sss, ssh, u, v], axis=0), dtype=torch.float32)
        
        layer_902 = self.ds_theta['thetao'].isel(time=idx, depth=self.idx_902).values
        layer_1062 = self.ds_theta['thetao'].isel(time=idx, depth=self.idx_1062).values
        layer_1000 = (self.w_1062 * layer_1062) + (self.w_902 * layer_902)
        
        mask = np.isnan(layer_1000)
        valid = layer_1000[~mask]
        
        if len(valid) > 0:
            mean, std = np.mean(valid), np.std(valid)
            if std == 0: std = 1
            layer_1000[~mask] = (layer_1000[~mask] - mean) / std
        layer_1000[mask] = 0
        
        target_tensor = torch.tensor(layer_1000, dtype=torch.float32).unsqueeze(0)
        return input_tensor, target_tensor

def train_1000m():
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    dataset = GLORYS_1000m_Dataset("temp_thetao_2026-08-30.nc")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = ResNet50_UNet_Pro(in_channels=5, out_channels=16).to(device)
    model.load_state_dict(torch.load("ocean_model_16_channel_stage1.pth", map_location=device, weights_only=True))
    
    for name, param in model.named_parameters():
        if "final_conv" not in name:
            param.requires_grad = False
            
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.MSELoss()
    
    orig_weight = model.final_conv.weight.data.clone()
    orig_bias = model.final_conv.bias.data.clone()
    
    # CRITICAL FIX: Use eval() to prevent batchnorm statistics (running mean/var) from 
    # being corrupted during this targeted 1000m fine-tuning pass!
    model.eval()
    
    print("Training Layer 15 (1000m) for 50 epochs...")
    for epoch in range(50):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            out_15 = outputs[:, 15:16, :, :]
            loss = criterion(out_15, targets)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.final_conv.weight[:15] = orig_weight[:15]
                model.final_conv.bias[:15] = orig_bias[:15]
                
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), "ocean_model_16_channel_stage2.pth")
    print("Saved ocean_model_16_channel_stage2.pth!")

if __name__ == "__main__":
    train_1000m()
