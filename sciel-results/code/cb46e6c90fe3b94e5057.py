import numpy as np


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.gamma = 0.85  # Aggressive forgetting for non-stationarity
        self.epsilon = 0.1  # Exploration rate
        self.ema = np.zeros(n_arms)  # EMA of rewards per arm
        self.last_played = np.zeros(n_arms, dtype=int)  # Last round each arm was played
        self.round = 0
        self.counts = np.zeros(n_arms)  # Total plays per arm

    def select(self, context):
        self.round += 1
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n))
        
        # For arms never played, give them a high initial estimate
        # Use UCB-like bonus: bonus = alpha * sqrt(log(t) / count)
        # But simpler: bonus based on time since last played
        alpha = 1.0
        bonus = np.zeros(self.n)
        for i in range(self.n):
            if self.counts[i] == 0:
                bonus[i] = 10.0  # Large bonus for unplayed arms
            else:
                # Bonus decays with time since last played
                time_since = self.round - self.last_played[i]
                bonus[i] = alpha * np.sqrt(np.log(self.round + 1) / self.counts[i])
        
        scores = self.ema + bonus
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        r = reward
        self.ema[arm] = self.gamma * self.ema[arm] + (1 - self.gamma) * r
        self.last_played[arm] = self.round
        self.counts[arm] += 1
