import urllib.request
import json
import urllib.error

url = 'http://127.0.0.1:8001/predict'
data = json.dumps({'date': '2026-08-30'}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as res:
        print(f"HTTP Status: {res.status}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(e.read().decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
