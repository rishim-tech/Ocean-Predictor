import copernicusmarine
import os
import argparse

def fetch_data(start_date, end_date, output_dir="training_data"):
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {
        "thetao": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "so": "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        "zos": "cmems_mod_glo_phy-zos_anfc_0.083deg_P1D-m",
        "cur": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
    }
    
    for var, dataset_id in datasets.items():
        out_file = os.path.join(output_dir, f"{var}_{start_date}_to_{end_date}.nc")
        print(f"Fetching {var} from {start_date} to {end_date}...")
        
        # Determine the target variable inside the dataset
        if var == "cur":
            target_vars = ["uo", "vo"]
        else:
            target_vars = [var]
            
        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=target_vars,
                start_datetime=f"{start_date}T00:00:00",
                end_datetime=f"{end_date}T00:00:00",
                minimum_longitude=40.0,
                maximum_longitude=100.0,
                minimum_latitude=-10.0,
                maximum_latitude=30.0,
                minimum_depth=0.0,
                maximum_depth=1100.0 if var != "zos" else 0.0,
                output_filename=out_file,
                force_download=True
            )
            print(f"Saved {out_file}")
        except Exception as e:
            print(f"Error fetching {var}: {e}")

if __name__ == "__main__":
    fetch_data("2024-01-01", "2024-01-02")
