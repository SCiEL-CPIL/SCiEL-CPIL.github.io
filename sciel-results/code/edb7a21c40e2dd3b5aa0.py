"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Context-free multi-armed bandit with EMA reward tracking."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.gamma = 0.90
        self.reward_ema = np.zeros(n_arms)
        self.reward_var_ema = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.c = 1.5
        self.round_count = 0

    def select(self, context):
        self.round_count += 1
        # For arms never pulled, use high exploration bonus
        scores = np.full(self.n, 1e6)
        for a in range(self.n):
            if self.counts[a] > 0:
                mean = self.reward_ema[a]
                var = self.reward_var_ema[a]
                # Use a simple UCB-like bonus based on variance and count
                bonus = self.c * np.sqrt(max(var, 0) / max(self.counts[a], 1))
                scores[a] = mean + bonus
            else:
                scores[a] = 1e6
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.counts[arm] += 1
        # Update reward EMA
        self.reward_ema[arm] = self.gamma * self.reward_ema[arm] + (1 - self.gamma) * reward
        # Update variance EMA
        diff = reward - self.reward_ema[arm]
        self.reward_var_ema[arm] = self.gamma * self.reward_var_ema[arm] + (1 - self.gamma) * (diff * diff)
# EVOLVE-BLOCK-END
