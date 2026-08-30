const axios = require('axios');

async function checkShape() {
  try {
    const res = await axios.post('http://127.0.0.1:8001/predict', { date: '2026-08-01' });
    const tensor = res.data.prediction_data;
    
    console.log("dim 1:", tensor?.length);
    console.log("dim 2:", tensor[0]?.length);
    console.log("dim 3:", tensor[0][0]?.length);
    console.log("dim 4:", tensor[0][0][0]?.length);
    console.log("type of tensor[0][0][0]:", typeof tensor[0][0][0]);
    
    // Check range & finite
    let min = Infinity, max = -Infinity, allFinite = true;
    
    function traverse(arr) {
        if (!Array.isArray(arr)) {
            if (arr !== null) {
                if (!Number.isFinite(arr)) allFinite = false;
                if (arr < min) min = arr;
                if (arr > max) max = arr;
            }
            return;
        }
        for (const item of arr) traverse(item);
    }
    traverse(tensor);
    console.log("min:", min, "max:", max, "finite:", allFinite);
  } catch (err) {
    console.log("Error:", err.message);
  }
}
checkShape();
