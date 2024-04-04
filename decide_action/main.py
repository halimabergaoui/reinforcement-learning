import numpy as np
import importlib.util
from flask import Flask, request, jsonify

def getAction(price_data):
    # Preset parameters
    model_path = '../data/models/Q_table.npy'
    reward_function_path = "../data/formulas/user-n.py"
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.95
    n_actions = 3  # Assuming 0: buy, 1: sell, 2: hold
    window_size = 50  # Assuming the last N prices including the new one

    # Function to choose an action
    def choose_action(state, epsilon, Q, n_actions):
        if np.random.rand() < epsilon:
            return np.random.choice(n_actions)  # Explore
        else:
            return np.argmax(Q[state])  # Exploit

    # Function to dynamically load the reward function
    def load_reward_function(reward_function_path):
        spec = importlib.util.spec_from_file_location("reward_function_module", reward_function_path)
        reward_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward_module)
        return reward_module.reward_function

    # Load the model (Q-table)
    Q = np.load(model_path)
    
    # Load the reward function
    reward_function = load_reward_function(reward_function_path)
    
    state = len(price_data) - window_size
    action = choose_action(state, epsilon, Q, n_actions)
    prices_window = price_data[-window_size:]
    reward = reward_function(action, prices_window, window_size)

    # Update Q-table
    Q[state, action] += alpha * (reward + gamma * np.max(Q[state]) - Q[state, action])
    
    # Save the updated model directly, overwriting the existing file
    np.save(model_path, Q)

    return action, reward

""" # Example usage
prices = [100, 101, 102, 103, 104]  # This needs to be a list of at least `window_size` number of prices
action, reward = getAction(prices)
print(f"Action taken: {action}, Reward received: {reward}") """

from flask import Flask, request, jsonify
import numpy as np
import importlib.util

app = Flask(__name__)


@app.route('/getPrice', methods=['POST'])
def get_price_api():
    # Get price data from the request
    data = request.get_json()
    price_data = data['price_data']
    
    # Check if price_data is valid
    if not price_data or not isinstance(price_data, list):
        return jsonify({'error': 'Invalid or missing price data'}), 400
    
    try:
        action, reward = getAction(price_data)
        action = int(action)
        reward = int(reward)


        return jsonify({'action': action, 'reward': reward})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

