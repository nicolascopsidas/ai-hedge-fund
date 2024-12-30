import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("POLYGON_API_KEY")
print(f"API Key: {api_key}")

url = f"https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/range/4/hour/2023-12-25/2023-12-29"
params = {
    "adjusted": "true",
    "sort": "asc",
    "apiKey": api_key
}

response = requests.get(url, params=params)
print(f"\nStatus Code: {response.status_code}")
print(f"Response: {response.text}")
