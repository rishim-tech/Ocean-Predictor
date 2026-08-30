import xarray as xr
import os

files = ["stats_sample.nc", "sample_depths.nc"]
for f in files:
    if os.path.exists(f):
        ds = xr.open_dataset(f)
        print(f"--- {f} ---")
        if "depth" in ds:
            print("Depths:", ds.depth.values)
        else:
            print("No depth dimension.")
