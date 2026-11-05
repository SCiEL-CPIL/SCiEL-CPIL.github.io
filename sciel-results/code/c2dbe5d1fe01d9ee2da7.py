# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary linear bandit with adaptive forgetting and UCB exploration."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        self.alpha = 1.0  # Exploration parameter
        self.gamma = 0.99  # Forgetting factor
        self.last_update_time = time.time()
        self.total_rounds = 0
        self.reward_history = [0.0 for _ in range(n_arms)]

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Solve for theta: A[a] @ theta = b[a]
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
            
            # Predicted reward
            pred = theta @ context
            
            # Uncertainty bonus
            try:
                # Compute context @ inv(A) @ context efficiently
                # inv(A) @ context is equivalent to solving A @ x = context
                x = np.linalg.solve(self.A[a], context)
                uncertainty = np.sqrt(context @ x)
            except np.linalg.LinAlgError:
                uncertainty = 0.0
            
            scores[a] = pred + self.alpha * uncertainty
        
        # Add small random noise for tie-breaking
        scores += np.random.randn(self.n) * 1e-6
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.total_rounds += 1
        
        # Update reward history for adaptive exploration
        self.reward_history[arm] = 0.95 * self.reward_history[arm] + 0.05 * reward
        
        # Adaptive forgetting: decrease gamma when rewards are volatile
        # This helps track non-stationarity
        if self.total_rounds > 10:
            # Estimate volatility from recent reward history
            vol = np.std(self.reward_history)
            # If volatility is high, use smaller gamma (faster forgetting)
            target_gamma = 0.995 - 0.01 * min(vol, 1.0)
            self.gamma = 0.9 * self.gamma + 0.1 * target_gamma
        
        # Update with forgetting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        self.count[arm] += 1
        
        # Adaptive exploration: decrease alpha over time but keep some exploration
        # Based on average reward magnitude
        avg_reward = np.mean([abs(r) for r in self.reward_history if r != 0]) if any(r != 0 for r in self.reward_history) else 1.0
        self.alpha = max(0.5, 1.0 * np.exp(-self.total_rounds / 200.0))
# EVOLVE-BLOCK-END
