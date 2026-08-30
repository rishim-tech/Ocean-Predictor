import copernicusmarine
import xarray as xr

def check_depths():
    dataset_id = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
    print(f"Fetching sample for dataset: {dataset_id}")
    try:
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=["thetao"],
            start_datetime="2024-01-01T00:00:00",
            end_datetime="2024-01-01T00:00:00",
            minimum_longitude=45.0,
            maximum_longitude=45.5,
            minimum_latitude=0.0,
            maximum_latitude=0.5,
            minimum_depth=0.0,
            maximum_depth=2000.0,
            output_filename="deep_sample.nc",
            force_download=True
        )
        ds = xr.open_dataset("deep_sample.nc")
        depths = ds.depth.values
        print("Native Depths found:")
        for d in depths:
            if d > 800:
                print(f"- {d} m")
    except Exception as e:
        print(f"Error checking {dataset_id}: {e}")

if __name__ == "__main__":
    check_depths()
