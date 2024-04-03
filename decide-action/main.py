import numpy as np
import importlib.util
import pandas as pd


# Dynamically load the reward function
spec = importlib.util.spec_from_file_location("module.name", "../data/formulas/user-n.py")
reward_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_module)

def choose_action(state, epsilon, Q, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

def update_model(prices, model_path, reward_function_path, epsilon, alpha, gamma, n_actions, window_size):
    # Load the model (Q-table)
    Q = np.load(model_path)
    
    # Dynamically load the reward function
    spec = importlib.util.spec_from_file_location("reward_function_module", reward_function_path)
    reward_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_module)
    reward_function = reward_module.reward_function

    # Assuming prices is a list of the latest prices including the window size
    state = len(prices) - window_size
    action = choose_action(state, epsilon, Q, n_actions)
    prices_window = prices[-window_size:]
    reward = reward_function(action, prices_window, window_size)

    # Update Q-table
    Q[state, action] += alpha * (reward - Q[state, action])
    
    # Save the updated model directly, overwriting the existing file
    np.save(model_path, Q)

    return action, reward, model_path
model_path = '../data/models/Q_table.npy'
reward_function_path = "../data/formulas/user-n.py"
epsilon = 0.1
alpha = 0.1
gamma = 0.95
n_actions = 3  # Assuming 0: buy, 1: sell, 2: hold
window_size = 50  # Assuming the last N prices including the new one
data = pd.read_csv('../data/prices/data.csv')
df = data[::-1].copy()  # Reversing the order if needed
prices = df['close'].values

# Split the dataset
split_ratio = 0.8
split_index = int(len(prices) * split_ratio)
train_prices = prices[:split_index]
test_prices = prices[split_index:]

action, reward, updated_model_path = update_model(test_prices, model_path, reward_function_path, epsilon, alpha, gamma, n_actions, window_size)
print(f"Action taken: {action}, Reward received: {reward}, Model updated at: {updated_model_path}")
