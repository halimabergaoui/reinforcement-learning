import numpy as np
import pandas as pd
import importlib.util

# Load the dataset
data = pd.read_csv('../data/prices/data.csv')
df = data[::-1].copy()  # Reversing the order if needed
prices = df['close'].values

# Split the dataset
split_ratio = 0.8
split_index = int(len(prices) * split_ratio)
train_prices = prices[:split_index]
test_prices = prices[split_index:]

# Parameters
gamma = 0.95
alpha = 0.1
epsilon = 0.1
n_episodes = 10
n_actions = 3  # 0: buy, 1: sell, 2: hold

# Initialize Q-table
Q = np.zeros((len(train_prices), n_actions))

def choose_action(state, epsilon, Q, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

# Dynamically load the reward_function
spec = importlib.util.spec_from_file_location("module.name", "../data/formulas/user-n.py")
reward_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_module)

def train(train_prices, Q, epsilon, alpha, gamma, n_episodes, n_actions, reward_function, window_size):
    if reward_function is None:
        raise ValueError("A reward function must be provided")
    
    print("Training...")
    for episode in range(n_episodes):
        episode_rewards = 0
        for index in range(window_size, len(train_prices)):  # Adjust for dynamic window
            state = index
            action = choose_action(state, epsilon, Q, n_actions)
            prices_window = train_prices[index-window_size+1:index+1]
            reward = reward_function(action, prices_window, window_size)
            next_state = state + 1 if index + 1 < len(train_prices) else state
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            episode_rewards += reward
        print(f"Episode {episode + 1}: Total Reward = {episode_rewards}")
    # Save the Q-table after training
    np.save('../data/models/Q_table.npy', Q)

def test(test_prices, Q, n_actions, reward_function, window_size):
    if reward_function is None:
        raise ValueError("A reward function must be provided")
    
    print("\nTesting...")
    test_rewards = 0
    for index in range(window_size, len(test_prices)):  # Adjust for dynamic window
        state = index
        action = np.argmax(Q[min(state, len(Q)-1)])
        prices_window = test_prices[index-window_size+1:index+1]
        reward = reward_function(action, prices_window, window_size)
        test_rewards += reward
    print(f"Total Reward in Testing = {test_rewards}")

# Load the Q-table if needed
# Q_loaded = np.load('/mnt/data/Q_table.npy')

# Example usage
window_size = 50  # This can now be dynamically adjusted
train(train_prices, Q, epsilon, alpha, gamma, n_episodes, n_actions, reward_module.reward_function, window_size)
test(test_prices, Q, n_actions, reward_module.reward_function, window_size)
