# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """LinUCB with a faster forgetting factor for non-stationary adaptation."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Faster decay to adapt to non-stationarity
        self.alpha = 1.0   # Exploration bonus coefficient
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

    def select(self, context):
        scores = []
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                pred = theta @ context
                # Calculate UCB bonus
                var = context @ inv @ context
                bonus = self.alpha * np.sqrt(var)
                scores.append(pred + bonus)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular (unlikely with eye init + decay)
                scores.append(np.random.randn())
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Decay existing statistics
        self.A[arm] = self.gamma * self.A[arm]
        self.b[arm] = self.gamma * self.b[arm]
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
# EVOLVE-BLOCK-END
