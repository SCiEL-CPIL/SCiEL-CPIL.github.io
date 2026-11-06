"""Non-stationary linear contextual bandit with global model and arm-identity features."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Global linear model with arm-identity features for non-stationary contextual bandit.
    
    Instead of per-arm models, we use a single global model where the feature vector
    is [context, arm_one_hot]. This allows:
    1. Sharing context representations across arms
    2. Learning inter-arm correlations
    3. Better sample efficiency in non-stationary settings
    
    We use Recursive Least Squares (RLS) with exponential forgetting for efficient
    online updates, and an adaptive exploration bonus.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 110.0  # Safety margin before 120s limit
        
        # Extended dimension: context + one-hot arm encoding
        self.d_ext = dim + n_arms
        
        # RLS state
        self.P = np.eye(self.d_ext) * 10.0  # Initial inverse covariance (large = high uncertainty)
        self.theta = np.zeros(self.d_ext)
        
        # Forgetting factor
        self.forgetting = 0.98
        
        # Track last play time for each arm for adaptive exploration
        self.last_played = np.full(n_arms, -1000)
        
        # Recent rewards for adaptive exploration
        self.recent_rewards = []

    def _build_feature(self, context, arm):
        """Build extended feature vector with arm identity."""
        x = np.zeros(self.d_ext)
        x[:self.d] = context
        x[self.d + arm] = 1.0
        return x

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return int(np.argmax(np.random.rand(self.n)))
        
        self.t += 1
        
        # Compute predicted rewards and exploration bonuses for all arms
        scores = np.zeros(self.n)
        for a in range(self.n):
            x = self._build_feature(context, a)
            # Predicted reward
            pred = self.theta @ x
            # Exploration bonus based on uncertainty and time since last play
            uncertainty = np.sqrt(x @ self.P @ x)
            time_since_play = self.t - self.last_played[a]
            # Adaptive exploration: more exploration for arms not played recently
            exploration_weight = 0.5 + 0.1 * np.log1p(time_since_play)
            scores[a] = pred + exploration_weight * uncertainty
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self.deadline:
            return
        
        self.t += 1
        self.last_played[arm] = self.t
        
        # Build feature vector
        x = self._build_feature(context, arm)
        
        # RLS update with forgetting
        # P = (1/forgetting) * (P - P @ x @ x.T @ P / (forgetting + x.T @ P @ x))
        P_x = self.P @ x
        denom = self.forgetting + x @ P_x
        gain = P_x / denom
        self.P = (self.P - np.outer(gain, P_x)) / self.forgetting
        self.theta += gain * (reward - self.theta @ x)
        
        # Track recent rewards for potential adaptive forgetting
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
# EVOLVE-BLOCK-END
