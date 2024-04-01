import pickle
import numpy as np
import pandas as pd

class QLearningTradingAgent:
    def __init__(self, prices, window_size=50, alpha=0.1, gamma=0.99):
        self.prices = prices
        self.window_size = window_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = {}  # Initialize Q-table as a dictionary
        self.actions = ['buy', 'sell', 'hold']
        self.total_rewards = []  # Log total rewards per episode

    def get_state(self, index):
        """Returns the state based on the window of prices."""
        if index - self.window_size >= -1:
            state = self.prices[max(0, index - self.window_size + 1):index + 1]
        else:
            state = np.zeros(self.window_size)
        # Simplification: using the sum of state as a discrete state representation
        return np.sum(state)

    def choose_action(self, state, epsilon):
        """Choose action based on an epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = [self.q_table.get((state, action), 0) for action in self.actions]
            max_q = max(q_values)
            # In case there're several actions with the same Q-value, choose one at random
            actions_with_max_q = [self.actions[i] for i, q in enumerate(q_values) if q == max_q]
            return np.random.choice(actions_with_max_q)

    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-table based on the action taken and the reward received."""
        next_max = max([self.q_table.get((next_state, a), 0) for a in self.actions])  # Max Q-value for the next state
        self.q_table[(state, action)] = self.q_table.get((state, action), 0) + \
                                        self.alpha * (reward + self.gamma * next_max - \
                                        self.q_table.get((state, action), 0))

    def get_reward(self, current_index, action):
        """Calculate reward based on the action and future price movement."""
        if current_index + 3 >= len(self.prices):
            return 0

        future_prices = self.prices[current_index + 1:current_index + 4]
        current_price = self.prices[current_index]
        sma = np.mean(self.prices[max(0, current_index - 49):current_index + 1])

        if action == 'sell' and all(current_price < p for p in future_prices):
            return 1
        elif action == 'buy' and all(current_price > sma for p in future_prices):
            return 1
        else:
            return -1  # Penalize wrong decisions or hold action

    def save_q_table(self, filename='q_table.pkl'):
        """Save the Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print("Q-table saved to", filename)

    def load_q_table(self, filename='q_table.pkl'):
        """Load the Q-table from a file."""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print("Q-table loaded from", filename)

    def train(self, episodes, epsilon=0.1):
        for e in range(episodes):
            state = self.get_state(0)
            total_reward = 0  # Reset total reward for the episode
            for t in range(len(self.prices) - 3):
                action = self.choose_action(state, epsilon)
                reward = self.get_reward(t, action)
                total_reward += reward  # Accumulate reward
                next_state = self.get_state(t + 1)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
            self.total_rewards.append(total_reward)  # Log total reward for the episode
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")
        
    def save_q_table(self, filename='q_table.pkl'):
        """Save the Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print("Q-table saved to", filename)

    def load_q_table(self, filename='q_table.pkl'):
        """Load the Q-table from a file."""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print("Q-table loaded from", filename)


    def test(self, test_prices):
        """Test the agent on a separate set of prices without updating the Q-table."""
        total_reward = 0
        for t in range(len(test_prices) - 3):
            state = self.get_state(t)
            action = self.choose_action(state, 0)  # Epsilon = 0 for testing (always choose best action)
            reward = self.get_reward(t, action)
            total_reward += reward
        print(f"Test Total Reward: {total_reward}")
        return total_reward

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    prices = df['close'].values

    # Split dataset into training and testing
    split_index = int(len(prices) * 0.8)
    train_prices = prices[:split_index]
    test_prices = prices[split_index:]

    agent = QLearningTradingAgent(train_prices)
    agent.train(episodes=100)
    
    # Save the Q-table
    agent.save_q_table('q_table.pkl')

    # To continue training or testing later, you can load the Q-table
    # agent.load_q_table('q_table.pkl')
    
    # Test the agent
    agent.test(test_prices)