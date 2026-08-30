import os
import numpy as np
import xarray as xr
import pandas as pd

def generate_mock_data():
    os.makedirs("training_data", exist_ok=True)
    
    # 15 existing + a few deep layers including 1062.44
    depths = [0.494, 5.078, 9.573, 18.496, 29.445, 47.374, 77.854, 92.326, 155.851, 186.126, 318.127, 380.213, 541.089, 643.567, 902.339, 1062.44, 1200.0]
    times = pd.date_range("2024-01-01", periods=2)
    lats = np.linspace(-10, 30, 480)
    lons = np.linspace(40, 100, 720)
    
    # thetao
    thetao_data = np.random.rand(len(times), len(depths), len(lats), len(lons)) * 30
    ds_thetao = xr.Dataset(
        {"thetao": (["time", "depth", "latitude", "longitude"], thetao_data)},
        coords={"time": times, "depth": depths, "latitude": lats, "longitude": lons}
    )
    ds_thetao.to_netcdf("training_data/thetao_mock.nc")
    
    # so
    so_data = np.random.rand(len(times), len(depths), len(lats), len(lons)) * 40
    ds_so = xr.Dataset(
        {"so": (["time", "depth", "latitude", "longitude"], so_data)},
        coords={"time": times, "depth": depths, "latitude": lats, "longitude": lons}
    )
    ds_so.to_netcdf("training_data/so_mock.nc")
    
    # zos
    zos_data = np.random.rand(len(times), len(lats), len(lons))
    ds_zos = xr.Dataset(
        {"zos": (["time", "latitude", "longitude"], zos_data)},
        coords={"time": times, "latitude": lats, "longitude": lons}
    )
    ds_zos.to_netcdf("training_data/zos_mock.nc")
    
    # cur (uo, vo)
    uo_data = np.random.rand(len(times), len(depths), len(lats), len(lons))
    vo_data = np.random.rand(len(times), len(depths), len(lats), len(lons))
    ds_cur = xr.Dataset(
        {"uo": (["time", "depth", "latitude", "longitude"], uo_data),
         "vo": (["time", "depth", "latitude", "longitude"], vo_data)},
        coords={"time": times, "depth": depths, "latitude": lats, "longitude": lons}
    )
    ds_cur.to_netcdf("training_data/cur_mock.nc")
    print("Mock data generated!")

if __name__ == "__main__":
    generate_mock_data()
