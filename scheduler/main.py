import requests
import schedule
import time

def fetch_solana_price():
    # Fetching Solana price from CoinGecko API
    coingecko_url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
    response = requests.get(coingecko_url)
    data = response.json()
    solana_price = data['solana']['usd']
    return solana_price

def send_price_to_api():
    solana_price = fetch_solana_price()
    print(f"Current Solana Price: ${solana_price}")

    # Assuming the second API requires a POST request with the price
    api_url = "https://example.com/api/action"  # Replace with the actual API URL
    payload = {'solana_price': solana_price}
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        print("Successfully sent the Solana price to the API.")
    else:
        print("Failed to send the Solana price to the API.")

# Schedule the task to run every day
schedule.every().day.at("10:00").do(send_price_to_api)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)
