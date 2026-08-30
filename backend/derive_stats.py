import copernicusmarine
import xarray as xr
import json
import numpy as np

print('Downloading Copernicus dataset for stats derivation...')
copernicusmarine.subset(
    dataset_id='cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m',
    variables=['thetao'],
    start_datetime='2024-01-01T00:00:00', end_datetime='2024-01-01T00:00:00',
    minimum_longitude=45.0, maximum_longitude=100.0,
    minimum_latitude=-10.0, maximum_latitude=30.0,
    minimum_depth=0.0, maximum_depth=1000.0,
    output_filename='stats_sample.nc',
    force_download=True
)

ds = xr.open_dataset('stats_sample.nc')
thetao = ds['thetao'].isel(time=0).values # shape: [depth, lat, lon]

depths = ds.depth.values
target_depths_meters = [0.5, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 925.0]
target_indices = []
for d in target_depths_meters:
    idx = (np.abs(depths - d)).argmin()
    target_indices.append(idx)

print('Target Indices:', target_indices)
print('Target Depths:', depths[target_indices])

means = []
stds = []
for i, idx in enumerate(target_indices):
    layer_data = thetao[idx]
    valid_data = layer_data[~np.isnan(layer_data)]
    if len(valid_data) > 0:
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        means.append(float(mean_val))
        stds.append(float(std_val))
    else:
        means.append(0.0)
        stds.append(1.0)

stats = {
    'mean': means,
    'std': stds,
    'indices': [int(x) for x in target_indices],
    'depths': [float(x) for x in depths[target_indices]]
}

with open('depth_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)
print('Saved depth_stats.json!')
