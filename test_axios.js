const axios = require('axios');

async function test() {
  try {
    const res = await axios.post('http://127.0.0.1:8001/predict', { date: '2026-08-01' });
    const data = res.data;
    const tensor = data.prediction_data;
    
    console.log("Response status:", res.status);
    console.log("Is array?", Array.isArray(tensor));
    console.log("tensor[0] is array?", Array.isArray(tensor[0]));
    console.log("tensor[0][0] is array?", Array.isArray(tensor[0][0]));
    console.log("tensor[0][0][0] is array?", Array.isArray(tensor[0][0][0]));
    
    if (
      !Array.isArray(tensor) ||
      !Array.isArray(tensor[0]) ||
      !Array.isArray(tensor[0][0])
    ) {
      throw new Error('Backend returned invalid prediction_data — expected a 4D tensor [batch, depth, lat, lon].');
    }
    
    console.log("SUCCESS!");
  } catch (err) {
    console.log("FAILED WITH ERROR:");
    console.log(err.message);
  }
}
test();
