import requests
import json

def test_inference():
    print("Testing /depths endpoint...")
    resp = requests.get("http://127.0.0.1:8000/depths")
    if resp.status_code == 200:
        depths = resp.json()["depths"]
        print(f"Depths returned: {depths}")
        if len(depths) != 16 or depths[-1] != 1000.0:
            print("ERROR: Depths do not match expected 16-channel 1000m target!")
        else:
            print("SUCCESS: Depths match!")
    else:
        print("ERROR: /depths failed", resp.text)
        
    print("\nTesting /predict endpoint (date=2024-01-01)...")
    resp = requests.post("http://127.0.0.1:8000/predict", json={"date": "2024-01-01"})
    if resp.status_code == 200:
        data = resp.json()
        print(f"Prediction successful! Target date: {data['date']}")
        print(f"Timings: {data['timings']}")
        
        preds = data["prediction_data"]
        # shape should be [1, 16, lat, lon]
        if len(preds) != 1 or len(preds[0]) != 16:
            print("ERROR: Prediction shape is wrong! Expected 16 channels.")
        else:
            print(f"SUCCESS: Returned prediction tensor with {len(preds[0])} depth layers!")
            
        # Check a specific pixel (e.g. at lat=200, lon=200) for the 1000m layer (index 15)
        # Note: frontend handles None for land mask
        sample = preds[0][15][200][200]
        print(f"Sample 1000m prediction value (denormalized): {sample} °C")
    else:
        print("ERROR: /predict failed", resp.text)

if __name__ == "__main__":
    test_inference()
