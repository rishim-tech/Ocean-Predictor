const axios = require('axios');

async function test() {
  try {
    const res = await axios.post('http://localhost:8001/predict', { date: '2026-08-01' });
    console.log("SUCCESS!");
  } catch (err) {
    console.log("FAILED WITH ERROR:");
    console.log(err.message);
  }
}
test();
