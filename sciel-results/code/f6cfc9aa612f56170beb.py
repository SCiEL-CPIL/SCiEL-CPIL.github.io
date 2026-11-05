# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with adaptive regularization
    and sliding-window recursive least squares (ARLS).
    
    Key features:
    1. Per-arm linear models with recursive least squares updates.
    2. Adaptive forgetting factor based on recent innovation variance.
    3. UCB exploration with time-adaptive exploration bonus.
    4. Deadline guard to ensure completion within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Per-arm sufficient statistics for RLS
        # A_inv: inverse of the regularized covariance matrix
        # b: weighted sum of rewards * contexts
        self.A_inv = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor
        self.base_forgetting = 0.99
        self.min_forgetting = 0.95
        self.max_forgetting = 0.999
        
        # Innovation tracking for adaptive forgetting
        self.innovation_sums = [0.0 for _ in range(n_arms)]
        self.innovation_counts = [0 for _ in range(n_arms)]
        self.last_rewards = [0.0 for _ in range(n_arms)]
        self.last_contexts = [None for _ in range(n_arms)]
        
        # Exploration parameters
        self.base_alpha = 1.0
        self.min_alpha = 0.5
        self.max_alpha = 3.0
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # Safety margin before 150s
        
        # Reward tracking for adaptive alpha
        self.recent_rewards = []
        self.reward_window = 50

    def _adaptive_forgetting(self, arm, innovation):
        """
        Compute adaptive forgetting factor based on recent innovation variance.
        High innovation -> lower forgetting (shorter memory)
        Low innovation -> higher forgetting (longer memory)
        """
        self.innovation_sums[arm] = 0.95 * self.innovation_sums[arm] + 0.05 * innovation ** 2
        self.innovation_counts[arm] = min(self.innovation_counts[arm] + 1, 1000)
        
        if self.innovation_counts[arm] < 10:
            return self.base_forgetting
        
        # Normalize innovation to [0, 1] range
        # Higher innovation variance -> need to forget faster
        norm_innovation = min(self.innovation_sums[arm] / 5.0, 1.0)
        
        # Map to forgetting factor: high innovation -> low forgetting
        forgetting = self.max_forgetting - norm_innovation * (self.max_forgetting - self.min_forgetting)
        return max(self.min_forgetting, min(self.max_forgetting, forgetting))

    def _adaptive_alpha(self):
        """
        Compute adaptive exploration bonus based on recent reward variance.
        """
        if len(self.recent_rewards) < 5:
            return self.base_alpha
        
        recent = self.recent_rewards[-self.reward_window:]
        reward_std = np.std(recent)
        reward_mean = np.mean(recent)
        
        # Higher variance -> more exploration
        norm_variance = min(reward_std / (abs(reward_mean) + 1.0), 1.0)
        alpha = self.min_alpha + norm_variance * (self.max_alpha - self.min_alpha)
        return alpha

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return arm with highest expected reward (greedy)
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.A_inv[a] @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)
        
        self.t += 1
        alpha = self._adaptive_alpha()
        
        scores = []
        for a in range(self.n):
            theta = self.A_inv[a] @ self.b[a]
            mu = theta @ context
            # UCB exploration term
            variance = context @ self.A_inv[a] @ context
            bonus = alpha * np.sqrt(variance)
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Check deadline
        if time.time() > self.deadline:
            return
        
        # Track reward for adaptive alpha
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.reward_window * 2:
            self.recent_rewards = self.recent_rewards[-self.reward_window:]
        
        # Compute innovation (prediction error)
        if self.last_contexts[arm] is not None:
            theta = self.A_inv[arm] @ self.b[arm]
            predicted = theta @ context
            innovation = reward - predicted
        else:
            innovation = reward
        
        # Adaptive forgetting factor
        forgetting = self._adaptive_forgetting(arm, innovation)
        
        # Update A_inv using Sherman-Morrison with forgetting
        # A_new = forgetting * A_old + x x^T
        # A_inv_new = (1/forgetting) * A_inv_old - (1/forgetting) * 
        #              (A_inv_old @ x @ x^T @ A_inv_old) / (forgetting + x^T @ A_inv_old @ x)
        
        Ax = self.A_inv[arm] @ context
        denom = forgetting + context @ Ax
        
        if abs(denom) > 1e-10:
            # Sherman-Morrison update with forgetting
            self.A_inv[arm] = (1.0 / forgetting) * self.A_inv[arm] - \
                              (1.0 / forgetting) * np.outer(Ax, Ax) / denom
        
        # Update b with forgetting
        self.b[arm] = forgetting * self.b[arm] + reward * context
        
        # Store for next innovation computation
        self.last_rewards[arm] = reward
        self.last_contexts[arm] = context.copy()
# EVOLVE-BLOCK-END
