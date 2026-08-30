import xarray as xr
ds = xr.open_dataset('temp_thetao_2026-08-30.nc')
print("2026-08-30 depths:", ds.depth.values)
try:
    ds2 = xr.open_dataset('temp_thetao_2024-01-01.nc')
    print("2024-01-01 depths:", ds2.depth.values)
except:
    pass
