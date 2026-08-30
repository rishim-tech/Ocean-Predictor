import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from main import ResNet50_UNet_Pro

def fix_weights():
    device = torch.device("cpu")
    model = ResNet50_UNet_Pro(in_channels=5, out_channels=16)
    
    print("Loading original weights from BEST_ocean_model_1YR (1).pth")
    state_dict = torch.load("BEST_ocean_model_1YR (1).pth", map_location=device, weights_only=True)
    new_state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        if name in new_state_dict:
            if name == "final_conv.weight":
                new_state_dict[name][:15] = param
            elif name == "final_conv.bias":
                new_state_dict[name][:15] = param
            else:
                new_state_dict[name].copy_(param)
                
    model.load_state_dict(new_state_dict)
    
    torch.save(model.state_dict(), "ocean_model_16_channel_stage1.pth")
    print("Fixed checkpoint saved!")

if __name__ == "__main__":
    fix_weights()
