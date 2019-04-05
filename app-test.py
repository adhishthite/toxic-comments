import requests

url = 'http://localhost:5000/predict'
payload = {'text': 'test_message'}
r = requests.get(url, params=payload)
print(r.json())