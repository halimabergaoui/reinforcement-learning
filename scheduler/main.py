import requests
import schedule
import time
from get_price import extract_prices  # Update this import according to your project structure

# Initialize the list to store rewards
all_rewards = []

def fetch_solana_price():
    # Read n_days from a file or define directly
    with open("../data/parameters/window_size", "r") as file:
        n_days = int(file.read().strip())
    print(n_days)    
    solana_prices = extract_prices(n_days)
    print(solana_prices)
    return solana_prices

def send_price_to_api():
    global all_rewards  # Access the global list of all rewards
    solana_prices = fetch_solana_price()
    solana_prices_list = [float(price) for price in solana_prices.tolist()]
    print(f"Current Solana Prices: {solana_prices_list}")

    api_url = "http://localhost:5000/getPrice"
    payload = {"price_data": solana_prices_list}

    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            action = response_data.get('action')
            reward = response_data.get('reward', 0)  # Default to 0 if no reward is present
            all_rewards.append(reward)  # Append the reward to the list
            sum_reward = sum(all_rewards) # Calculate the average reward
            length_reward = len(all_rewards) 
            print(f"Action: {action}, Reward: {reward}, Average Reward: {sum_reward}/{length_reward}")
        else:
            print(f"Error from server: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# First call to send_price_to_api to start the process
send_price_to_api()

# Schedule the task to run every 10 minutes (use .minutes for actual use, here it's .seconds for illustration)
schedule.every(10).seconds.do(send_price_to_api)  # Adjusted for demonstration purposes

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)
