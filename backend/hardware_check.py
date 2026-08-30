import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import psutil
import json

def get_hardware_info():
    info = {
        "RAM_GB": psutil.virtual_memory().total / (1024**3),
        "CPU_cores": psutil.cpu_count(logical=True),
        "GPU_available": torch.cuda.is_available()
    }
    if info["GPU_available"]:
        info["GPU_name"] = torch.cuda.get_device_name(0)
        info["GPU_VRAM_GB"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        info["GPU_name"] = "None"
        info["GPU_VRAM_GB"] = 0
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    get_hardware_info()
