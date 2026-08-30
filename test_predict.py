import urllib.request
import time
import json

try:
    t0 = time.time()
    req = urllib.request.Request(
        'http://127.0.0.1:8001/predict', 
        data=b'{"date": "2026-08-01"}', 
        headers={'Content-Type': 'application/json'}
    )
    res = urllib.request.urlopen(req)
    print(res.getcode())
    print(res.read().decode('utf-8')[:300]) 
    print(time.time() - t0)
except urllib.error.HTTPError as e:
    print(e.code)
    print(e.read().decode('utf-8'))
except Exception as e:
    print(e)
