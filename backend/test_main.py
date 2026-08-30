import asyncio
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from main import app, PredictRequest
from fastapi.testclient import TestClient

def main():
    client = TestClient(app)
    print("Testing /depths")
    res = client.get("/depths")
    print(res.status_code)
    print(res.json())

    print("\nTesting /predict")
    res = client.post("/predict", json={"date": "2026-08-30"})
    print(res.status_code)
    data = res.json()
    if "prediction_data" in data:
        tensor = data["prediction_data"]
        # shape [1, 15, lat, lon]
        layers = tensor[0]
        for i, layer in enumerate(layers):
            # Compute mean of non-null values
            valid = [v for row in layer for v in row if v is not None]
            if len(valid) > 0:
                mean_v = sum(valid)/len(valid)
                min_v = min(valid)
                max_v = max(valid)
                print(f"Layer {i}: mean={mean_v:.2f}, min={min_v:.2f}, max={max_v:.2f}")
            else:
                print(f"Layer {i}: all null")
    else:
        print(data)

if __name__ == "__main__":
    main()
