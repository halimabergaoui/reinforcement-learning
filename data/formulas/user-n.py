import numpy as np
def reward_function(action, prices_window, window_size):
    #print(len(prices_window),window_size)
    if len(prices_window) < window_size:
        return 0  # Adjusted to account for dynamic window size
    sma = np.mean(prices_window[-window_size:])
    reward = 0
    if action == 1 and all(p < sma for p in prices_window[-10:]):
        reward = 1
    elif action == 0 and all(p > sma for p in prices_window[-10:]):
        reward = 1
    return reward