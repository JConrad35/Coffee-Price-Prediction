import requests

API_KEY = "AN4ZUH0YE0YDLWQ4"  # Keep this private

def get_coffee_price():
    """Fetches monthly coffee price data from Alpha Vantage API."""
    url = f"https://www.alphavantage.co/query?function=COFFEE&interval=monthly&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    return data