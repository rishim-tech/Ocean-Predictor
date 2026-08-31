import json
import time
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import math
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
    # in_channels=7: SST, SSS, SSH, U_cur, V_cur, U_wind, V_wind
    # out_channels=15: 15 native GLORYS depth levels (0.5m – 902m)
    def __init__(self, in_channels=7, out_channels=15):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Seed all in_channels slots cyclically from the 3 pretrained RGB channels.
            # In practice load_state_dict() overwrites conv1.weight with the trained checkpoint.
            for c in range(in_channels):
                self.conv1.weight[:, c:c + 1, :, :] = resnet.conv1.weight[:, c % 3:c % 3 + 1, :, :]
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "BEST_ocean_model_1YR_7CH.pth")
model = ResNet50_UNet_Pro(in_channels=7, out_channels=15).to(device)

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
means = torch.tensor(stats["mean"][:15], dtype=torch.float32).view(1, 15, 1, 1).to(device)
stds = torch.tensor(stats["std"][:15], dtype=torch.float32).view(1, 15, 1, 1).to(device)

# All 16 depth levels served to the frontend (15 native + derived 1000m)
MODEL_DEPTHS = stats["depths"]

# ==========================================
# 3.6 PREDICTION CACHE
# ==========================================
# In-memory cache keyed by date string.
# Each entry is a dict with:
#   "prediction_data": the serialized prediction (nested list),
#   "input_fields":    dict of 7 serialized 2D arrays (nested lists)
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
            "prediction_data": cached["prediction_data"],
            "input_fields": cached["input_fields"],
            "timings": timings,
        }
    
    timings["cache_hit"] = False
    
    # Temporary filenames for Copernicus downloads
    file_thetao = f"temp_thetao_{target_date}.nc"
    file_so = f"temp_so_{target_date}.nc"
    file_cur = f"temp_cur_{target_date}.nc"
    file_zos = f"temp_zos_{target_date}.nc"
    file_wind = f"temp_wind_{target_date}.nc"
    
    t_fetch_start = time.time()
    
    try:
        # Check if files already exist in cache (skip download if they do)
        cache_hit_disk = True
        for f in [file_thetao, file_so, file_cur, file_zos, file_wind]:
            if not os.path.exists(f) or os.path.getsize(f) < 1000:
                cache_hit_disk = False
                break
                
        if cache_hit_disk:
            print(f"Using cached Copernicus NetCDF files for {target_date}...")
            # We still record fetch time (very fast)
        else:
            print(f"Downloading live satellite data for {target_date} (5 streams in parallel)...")

            def fetch_thetao():
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
                    variables=["thetao"],
                    start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                    minimum_longitude=45.0, maximum_longitude=100.0,
                    minimum_latitude=-10.0, maximum_latitude=30.0,
                    minimum_depth=0.49402499198913574,
                    maximum_depth=0.49402499198913574,
                    coordinates_selection_method="nearest",
                    output_filename=file_thetao,
                    force_download=True
                )

            def fetch_so():
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
                    variables=["so"],
                    start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                    minimum_longitude=45.0, maximum_longitude=100.0,
                    minimum_latitude=-10.0, maximum_latitude=30.0,
                    minimum_depth=0.49402499198913574,
                    maximum_depth=0.49402499198913574,
                    coordinates_selection_method="nearest",
                    output_filename=file_so,
                    force_download=True
                )

            def fetch_cur():
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
                    variables=["uo", "vo"],
                    start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                    minimum_longitude=45.0, maximum_longitude=100.0,
                    minimum_latitude=-10.0, maximum_latitude=30.0,
                    minimum_depth=0.49402499198913574,
                    maximum_depth=0.49402499198913574,
                    coordinates_selection_method="nearest",
                    output_filename=file_cur,
                    force_download=True
                )

            def fetch_zos():
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                    variables=["zos"],
                    start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                    minimum_longitude=45.0, maximum_longitude=100.0,
                    minimum_latitude=-10.0, maximum_latitude=30.0,
                    output_filename=file_zos,
                    force_download=True
                )

            def fetch_wind():
                """Try requested date, fall back to previous day if NRT lag."""
                for attempt_date in [target_date, (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')]:
                    try:
                        copernicusmarine.subset(
                            dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
                            variables=["eastward_wind", "northward_wind"],
                            start_datetime=f"{attempt_date}T00:00:00", end_datetime=f"{attempt_date}T23:59:59",
                            minimum_longitude=45.0, maximum_longitude=100.0,
                            minimum_latitude=-10.0, maximum_latitude=30.0,
                            coordinates_selection_method="nearest",
                            output_filename=file_wind,
                            force_download=True
                        )
                        if attempt_date != target_date:
                            print(f"  ⚠ Wind data used fallback date {attempt_date}")
                        return
                    except Exception:
                        continue
                # If both dates fail, write a sentinel so preprocessing knows
                with open(file_wind, 'w') as f:
                    f.write('NO_WIND_DATA')
                print("  ⚠ Wind data unavailable — using zero-fill")

            # Run all 5 downloads concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(fetch_thetao): "thetao",
                    executor.submit(fetch_so): "salinity",
                    executor.submit(fetch_cur): "currents",
                    executor.submit(fetch_zos): "ssh",
                    executor.submit(fetch_wind): "wind",
                }
                for future in as_completed(futures):
                    name = futures[future]
                    future.result()  # Raise if any download failed
                    print(f"  ✓ {name} downloaded")
        
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

        # Wind: load if available, otherwise zero-fill
        wind_available = False
        try:
            ds_wind = xr.open_dataset(file_wind)
            wind_available = True
        except Exception:
            pass

        if wind_available:
            # Wind is hourly -> average across the day's timesteps to get a
            # daily-resolution field, then regrid onto the same lat/lon grid
            # as the other variables (wind product is on a 0.125° grid,
            # the ocean variables are on a 0.083° grid).
            u_wind_raw = ds_wind['eastward_wind'].mean(dim='time')
            v_wind_raw = ds_wind['northward_wind'].mean(dim='time')
            u_wind_raw = u_wind_raw.interp(
                latitude=sst_raw['latitude'], longitude=sst_raw['longitude'], method='nearest'
            )
            v_wind_raw = v_wind_raw.interp(
                latitude=sst_raw['latitude'], longitude=sst_raw['longitude'], method='nearest'
            )
        
        # Land mask is based on NaNs in the CMEMS SST data
        mask = np.isnan(sst_raw.values)
        
        sst = sst_raw.fillna(0).values
        sss = sss_raw.fillna(0).values
        u_cur = u_raw.fillna(0).values
        v_cur = v_raw.fillna(0).values
        ssh = ssh_raw.fillna(0).values
        if wind_available:
            u_wind = u_wind_raw.fillna(0).values
            v_wind = v_wind_raw.fillna(0).values
        else:
            u_wind = np.zeros_like(sst)
            v_wind = np.zeros_like(sst)
        
        # Build 7-channel input tensor: SST, SSS, SSH, U_cur, V_cur, U_wind, V_wind
        input_array = np.stack([sst, sss, ssh, u_cur, v_cur, u_wind, v_wind], axis=0)
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
        
        timings["preprocess_s"] = round(time.time() - t_preprocess, 3)
        
        # ── T10: Model inference ────────────────────────────
        t_inference = time.time()
        
        print("Running AI Prediction...")
        with torch.no_grad():
            # Pad input to a multiple of 32 to prevent UNet edge interpolation artifacts
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            pad_h = (32 - (h % 32)) % 32
            pad_w = (32 - (w % 32)) % 32
            if pad_h > 0 or pad_w > 0:
                # Replicate padding on right and bottom to avoid sharp gradients
                input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='replicate')

            prediction = model(input_tensor)  # Shape: (1, 15, H_padded, W_padded)
            
            # Crop back to original requested resolution
            if pad_h > 0 or pad_w > 0:
                prediction = prediction[:, :, :h, :w]

            # The model natively outputs raw °C for all 15 channels. 
            # Do NOT denormalize any channels, as this corrupts the physical values.

            # Derive 16th channel (1000m) by linear extrapolation from
            # the two deepest native layers: idx 13 (643.57m) and idx 14 (902.34m)
            d13 = MODEL_DEPTHS[13]  # 643.57 m
            d14 = MODEL_DEPTHS[14]  # 902.34 m
            slope = (prediction[:, 14:15, :, :] - prediction[:, 13:14, :, :]) / (d14 - d13)
            pred_1000 = prediction[:, 14:15, :, :] + slope * (1000.0 - d14)
            prediction = torch.cat([prediction, pred_1000], dim=1)  # Shape: (1, 16, H, W)
        
        timings["inference_s"] = round(time.time() - t_inference, 3)
        
        # ── T11-T12: Postprocessing + serialization ─────────
        t_post = time.time()
        
        pred_numpy = prediction.squeeze(0).cpu().numpy()  # Shape: (16, lat, lon)
        
        # ── INTERPOLATE TO STANDARD DEPTHS ──────────────────
        # Problem Statement Requirement: Reconstruct temperature at standard depth levels
        # (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000)
        from scipy.interpolate import interp1d
        STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        
        # Interpolate along the depth axis (axis 0)
        interpolator = interp1d(MODEL_DEPTHS, pred_numpy, axis=0, fill_value="extrapolate")
        pred_numpy = interpolator(STANDARD_DEPTHS) # Shape becomes (15, lat, lon)
        
        # Apply land mask: SST == 0.0 is the CMEMS land mask filled with 0
        land_mask = (sst == 0.0)
        pred_numpy[:, land_mask] = np.nan
        
        # Round to 2 decimal places to reduce JSON payload size
        pred_rounded = np.round(pred_numpy, 2)
        
        # Convert np.nan to None so that it serializes to 'null' in JSON
        # This prevents fake prediction data over land
        pred_rounded = np.where(np.isnan(pred_rounded), None, pred_rounded)
        pred_list = [pred_rounded.tolist()]
        
        # ── Serialize the 7 raw input fields (same land-mask / NaN→None treatment) ──
        _input_field_names = ["sst", "sss", "ssh", "u_cur", "v_cur", "u_wind", "v_wind"]
        _input_field_arrays = [sst, sss, ssh, u_cur, v_cur, u_wind, v_wind]
        input_fields = {}
        for name, arr in zip(_input_field_names, _input_field_arrays):
            arr_copy = arr.copy().astype(float)
            arr_copy[land_mask] = np.nan
            arr_rounded = np.round(arr_copy, 2)
            arr_safe = np.where(np.isnan(arr_rounded), None, arr_rounded)
            input_fields[name] = arr_safe.tolist()
        
        timings["postprocess_s"] = round(time.time() - t_post, 3)
        
        # ── Cache the result ────────────────────────────────
        _prediction_cache[cache_key] = {
            "prediction_data": pred_list,
            "input_fields": input_fields,
        }
        _evict_if_needed()
        
        # ── Cleanup is handled in finally block ─────────────
        
        timings["total_s"] = round(time.time() - t_start, 3)
        print(f"Prediction complete in {timings['total_s']}s "
              f"(fetch={timings['fetch_s']}s, preprocess={timings['preprocess_s']}s, "
              f"inference={timings['inference_s']}s, postprocess={timings['postprocess_s']}s)")
            
        return {
            "status": "success",
            "date": target_date,
            "depths": STANDARD_DEPTHS,
            "prediction_data": pred_list,
            "input_fields": input_fields,
            "timings": timings,
        }
        
    except HTTPException:
        raise  # Re-raise explicit HTTP exceptions (e.g. 401 Unauthorized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Close xarray datasets if they were opened to release file locks
        for ds_name in ['ds_thetao', 'ds_so', 'ds_cur', 'ds_zos', 'ds_wind']:
            if ds_name in locals() and hasattr(locals()[ds_name], 'close'):
                try: locals()[ds_name].close()
                except: pass
                
        # Note: We intentionally do NOT delete the temp .nc files here 
        # so they can serve as a persistent local cache for subsequent identical requests.


# ==========================================
# 5. INPUT TRANSPARENCY ENDPOINT
# ==========================================
@app.get("/inputs/{date}")
def get_inputs(date: str):
    """Return the 7 raw input fields for a previously-predicted date.
    Only works for dates already in the in-memory cache — does NOT
    trigger a new Copernicus download.
    """
    cache_key = _cache_key(date)
    if cache_key not in _prediction_cache:
        return {
            "status": "not_found",
            "reason": f"No cached prediction for date '{date}'. Run /predict first.",
        }
    cached = _prediction_cache[cache_key]
    return {
        "status": "success",
        "date": date,
        "fields": ["sst", "sss", "ssh", "u_cur", "v_cur", "u_wind", "v_wind"],
        "input_fields": cached["input_fields"],
    }


# ==========================================
# 6. ARGO VALIDATION / SKILL METRICS
# ==========================================

def load_argo_for_date(date_str):
    """Load ARGO profile observations for a given date.

    TODO: Replace this placeholder with real loading logic.
    Should return a numpy array of shape (n_depths, lat, lon) — or a
    structured array of point-pairs — containing observed temperature
    at the same MODEL_DEPTHS levels.  Grid cells with no observation
    should be np.nan.

    Parameters
    ----------
    date_str : str
        Date in "YYYY-MM-DD" format.

    Returns
    -------
    np.ndarray
        Array of shape (len(MODEL_DEPTHS), lat, lon) with observed
        temperatures, NaN where no ARGO data is available.

    Raises
    ------
    NotImplementedError
        Always, until the real loader is wired in.
    """
    raise NotImplementedError(
        f"ARGO data loader not yet implemented. "
        f"Provide a function that returns a (n_depths, lat, lon) numpy array "
        f"of observed temperatures for date '{date_str}', with NaN for missing cells."
    )


def compute_skill_metrics(prediction, argo_obs, depths):
    """Compute per-depth Pearson correlation, RMSE, and Bias.

    Parameters
    ----------
    prediction : np.ndarray
        Model prediction, shape (n_depths, lat, lon).
    argo_obs : np.ndarray
        ARGO observations, same shape. NaN where no observation.
    depths : list[float]
        Depth values corresponding to axis-0.

    Returns
    -------
    dict with keys "depths", "correlation", "rmse", "bias" — each a
    list of len(depths) floats (or None where insufficient data).
    """
    n_depths = len(depths)
    correlations = []
    rmses = []
    biases = []

    for d in range(n_depths):
        pred_slice = prediction[d].ravel().astype(float)
        obs_slice = argo_obs[d].ravel().astype(float)

        # Keep only cells where BOTH pred and obs are valid (non-NaN)
        valid = np.isfinite(pred_slice) & np.isfinite(obs_slice)
        n_valid = int(np.sum(valid))

        if n_valid < 2:
            # Not enough points for meaningful statistics
            correlations.append(None)
            rmses.append(None)
            biases.append(None)
            continue

        p = pred_slice[valid]
        o = obs_slice[valid]
        diff = p - o

        # Bias = mean(pred - obs)
        bias = float(np.mean(diff))

        # RMSE = sqrt(mean((pred - obs)^2))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        # Pearson correlation — guard against zero-variance
        p_std = np.std(p)
        o_std = np.std(o)
        if p_std < 1e-12 or o_std < 1e-12:
            corr = None
        else:
            corr_matrix = np.corrcoef(p, o)
            r = float(corr_matrix[0, 1])
            corr = None if (math.isnan(r) or math.isinf(r)) else round(r, 6)

        correlations.append(corr)
        rmses.append(round(rmse, 4))
        biases.append(round(bias, 4))

    return {
        "depths": depths,
        "correlation": correlations,
        "rmse": rmses,
        "bias": biases,
    }


@app.get("/skill/{date}")
def get_skill_metrics(date: str):
    """Compute skill metrics (correlation, RMSE, bias) per depth level
    between the model's cached prediction and independent ARGO observations.
    """
    # 1. Check that we have a cached prediction for this date
    cache_key = _cache_key(date)
    if cache_key not in _prediction_cache:
        return {
            "status": "unavailable",
            "reason": f"No cached prediction for date '{date}'. Run /predict first.",
        }

    # 2. Try to load ARGO observations
    try:
        argo_obs = load_argo_for_date(date)
    except NotImplementedError as e:
        return {
            "status": "unavailable",
            "reason": str(e),
        }
    except FileNotFoundError:
        return {
            "status": "unavailable",
            "reason": f"ARGO observation file not found for date '{date}'.",
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "reason": f"Failed to load ARGO data: {e}",
        }

    # 3. Recover the prediction as a numpy array from the cache
    #    pred_list is [[depth0_rows, depth1_rows, ...]] with None for land
    try:
        cached = _prediction_cache[cache_key]
        pred_nested = cached["prediction_data"][0]  # list of 16 depth layers
        # Convert back to numpy, restoring None -> NaN
        pred_array = np.array(pred_nested, dtype=float)  # NaN where None was
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to deserialize cached prediction: {e}",
        }

    # 4. Compute metrics
    try:
        metrics = compute_skill_metrics(pred_array, argo_obs, MODEL_DEPTHS)
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Metric computation failed: {e}",
        }

    return {
        "status": "success",
        "date": date,
        **metrics,
    }
