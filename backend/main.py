import json
import time
import hashlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import copernicusmarine
import xarray as xr

# ==========================================
# 1. INITIALIZE FASTAPI APP
# ==========================================
app = FastAPI(title="Ocean AI Backend - Live Fetch")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Data Model
class PredictRequest(BaseModel):
    date: str  # Format: "YYYY-MM-DD"

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
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
    def __init__(self, in_channels=5, out_channels=16):
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

# ==========================================
# 3. LOAD MODEL 
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting Backend... Using device: {device}")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ocean_model_16_channel_stage2.pth")
model = ResNet50_UNet_Pro(in_channels=5, out_channels=16).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval() 
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")


# ==========================================
# 3.5 LOAD DENORMALIZATION STATS
# ==========================================
STATS_PATH = os.path.join(os.path.dirname(__file__), "depth_stats.json")
with open(STATS_PATH, "r") as f:
    stats = json.load(f)
means = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 16, 1, 1).to(device)
stds = torch.tensor(stats["std"], dtype=torch.float32).view(1, 16, 1, 1).to(device)

# The 15 native GLORYS depth levels the model was trained on (meters)
MODEL_DEPTHS = stats["depths"]

# ==========================================
# 3.6 PREDICTION CACHE
# ==========================================
# In-memory cache keyed by date string.
# Stores the full denormalized prediction numpy array.
# This avoids repeated Copernicus downloads for the same date.
_prediction_cache = {}
_MAX_CACHE_ENTRIES = 5  # Keep up to 5 dates cached


def _cache_key(date_str):
    """Generate a cache key from the date string."""
    return date_str


def _evict_if_needed():
    """Evict oldest entry if cache exceeds max size."""
    while len(_prediction_cache) > _MAX_CACHE_ENTRIES:
        oldest_key = next(iter(_prediction_cache))
        del _prediction_cache[oldest_key]


# ==========================================
# 4. API ROUTES & LIVE PROCESSING
# ==========================================
@app.get("/")
def read_root():
    return {"message": "Ocean Predictor Live Backend is UP!"}

@app.get("/depths")
def get_depths():
    """Return the model's native depth levels so the frontend can stay in sync."""
    return {"depths": MODEL_DEPTHS}

@app.post("/predict")
async def predict_subsurface(request: PredictRequest):
    target_date = request.date
    timings = {}
    t_start = time.time()
    
    # ── Cache check ──────────────────────────────────────
    cache_key = _cache_key(target_date)
    if cache_key in _prediction_cache:
        cached = _prediction_cache[cache_key]
        timings["cache_hit"] = True
        timings["total_s"] = round(time.time() - t_start, 3)
        print(f"Cache HIT for {target_date} - returning in {timings['total_s']}s")
        return {
            "status": "success",
            "date": target_date,
            "depths": MODEL_DEPTHS,
            "prediction_data": cached,
            "timings": timings,
        }
    
    timings["cache_hit"] = False
    
    # Temporary filenames for Copernicus downloads
    file_thetao = f"temp_thetao_{target_date}.nc"
    file_so = f"temp_so_{target_date}.nc"
    file_cur = f"temp_cur_{target_date}.nc"
    file_zos = f"temp_zos_{target_date}.nc"
    
    t_fetch_start = time.time()
    
    try:
        # Check if files already exist in cache (skip download if they do)
        cache_hit_disk = True
        for f in [file_thetao, file_so, file_cur, file_zos]:
            if not os.path.exists(f) or os.path.getsize(f) < 1000:
                cache_hit_disk = False
                break
                
        if cache_hit_disk:
            print(f"Using cached Copernicus NetCDF files for {target_date}...")
            # We still record fetch time (very fast)
        else:
            print(f"Downloading live satellite data for {target_date} (4 streams)...")
            # 1. Fetch Temperature (thetao)
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
                variables=["thetao"],
                start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                minimum_longitude=45.0, maximum_longitude=100.0,
                minimum_latitude=-10.0, maximum_latitude=30.0,
                minimum_depth=0.4, maximum_depth=0.5,
                output_filename=file_thetao,
                force_download=True
            )
            
            # 2. Fetch Salinity (so)
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
                variables=["so"],
                start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                minimum_longitude=45.0, maximum_longitude=100.0,
                minimum_latitude=-10.0, maximum_latitude=30.0,
                minimum_depth=0.4, maximum_depth=0.5,
                output_filename=file_so,
                force_download=True
            )

            # 3. Fetch Currents (uo, vo)
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
                variables=["uo", "vo"],
                start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                minimum_longitude=45.0, maximum_longitude=100.0,
                minimum_latitude=-10.0, maximum_latitude=30.0,
                minimum_depth=0.4, maximum_depth=0.5,
                output_filename=file_cur,
                force_download=True
            )

            # 4. Fetch Sea Surface Height (zos) -> 2D Variable, depth is omitted
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                variables=["zos"],
                start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                minimum_longitude=45.0, maximum_longitude=100.0,
                minimum_latitude=-10.0, maximum_latitude=30.0,
                output_filename=file_zos,
                force_download=True
            )
        
        timings["fetch_s"] = round(time.time() - t_fetch_start, 3)

        # ── T5-T6: NetCDF open + preprocessing ─────────────
        t_preprocess = time.time()
        
        ds_thetao = xr.open_dataset(file_thetao)
        ds_so = xr.open_dataset(file_so)
        ds_cur = xr.open_dataset(file_cur)
        ds_zos = xr.open_dataset(file_zos)
        
        sst_raw = ds_thetao['thetao'].isel(time=0, depth=0)
        sss_raw = ds_so['so'].isel(time=0, depth=0)
        u_raw = ds_cur['uo'].isel(time=0, depth=0)
        v_raw = ds_cur['vo'].isel(time=0, depth=0)
        ssh_raw = ds_zos['zos'].isel(time=0)
        
        # Land mask is based on NaNs in the CMEMS SST data
        mask = np.isnan(sst_raw.values)
        
        sst = sst_raw.fillna(0).values
        sss = sss_raw.fillna(0).values
        u_cur = u_raw.fillna(0).values
        v_cur = v_raw.fillna(0).values
        ssh = ssh_raw.fillna(0).values
        
        # Build 5-channel input tensor
        input_array = np.stack([sst, sss, ssh, u_cur, v_cur], axis=0)
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
        
        timings["preprocess_s"] = round(time.time() - t_preprocess, 3)
        
        # ── T10: Model inference ────────────────────────────
        t_inference = time.time()
        
        print("Running AI Prediction...")
        with torch.no_grad():
            prediction = model(input_tensor)
            # FIX: The original model was inconsistently trained. 
            # Channels 0, 14 (and our new 15) output normalized data.
            # Channels 1-13 output raw unnormalized temperature.
            # We ONLY denormalize the normalized channels to prevent double-scaling (68+ °C bug)
            for c in range(16):
                if c in [0, 14, 15]:
                    prediction[:, c, :, :] = (prediction[:, c, :, :] * stds[:, c, :, :]) + means[:, c, :, :]
        
        timings["inference_s"] = round(time.time() - t_inference, 3)
        
        # ── T11-T12: Postprocessing + serialization ─────────
        t_post = time.time()
        
        pred_numpy = prediction.squeeze(0).cpu().numpy() # Shape: (15, lat, lon)
        
        # Apply land mask: SST == 0.0 is the CMEMS land mask filled with 0
        land_mask = (sst == 0.0)
        pred_numpy[:, land_mask] = np.nan
        
        # Round to 2 decimal places to reduce JSON payload size
        pred_rounded = np.round(pred_numpy, 2)
        
        # Convert np.nan to None so that it serializes to 'null' in JSON
        # This prevents fake prediction data over land
        pred_rounded = np.where(np.isnan(pred_rounded), None, pred_rounded)
        pred_list = [pred_rounded.tolist()]
        
        timings["postprocess_s"] = round(time.time() - t_post, 3)
        
        # ── Cache the result ────────────────────────────────
        _prediction_cache[cache_key] = pred_list
        _evict_if_needed()
        
        # ── Cleanup is handled in finally block ─────────────
        
        timings["total_s"] = round(time.time() - t_start, 3)
        print(f"Prediction complete in {timings['total_s']}s "
              f"(fetch={timings['fetch_s']}s, preprocess={timings['preprocess_s']}s, "
              f"inference={timings['inference_s']}s, postprocess={timings['postprocess_s']}s)")
            
        return {
            "status": "success",
            "date": target_date,
            "depths": MODEL_DEPTHS,
            "prediction_data": pred_list,
            "timings": timings,
        }
        
    except HTTPException:
        raise  # Re-raise explicit HTTP exceptions (e.g. 401 Unauthorized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Close xarray datasets if they were opened to release file locks
        for ds_name in ['ds_thetao', 'ds_so', 'ds_cur', 'ds_zos']:
            if ds_name in locals() and hasattr(locals()[ds_name], 'close'):
                try: locals()[ds_name].close()
                except: pass
                
        # Note: We intentionally do NOT delete the temp .nc files here 
        # so they can serve as a persistent local cache for subsequent identical requests.
