"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Policy:
    """Discounted Linear Bandit."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Discount factor for non-stationarity
        self.beta = 1.5    # Exploration parameter
        # Initialize A (design matrix) and b (reward vector) for each arm
        # A is d x d, b is d x 1
        self.A = [np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        self.t = 0

    def select(self, context):
        context = np.asarray(context).flatten()
        scores = []
        for a in range(self.n):
            # Compute theta = A^{-1} b
            try:
                A_inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                # Fallback if singular (unlikely with I init and gamma < 1)
                A_inv = np.eye(self.d)
            
            theta = A_inv @ self.b[a]
            pred = theta @ context
            
            # Compute variance: context^T A^{-1} context
            var = context @ A_inv @ context
            
            # UCB score
            score = pred + self.beta * np.sqrt(max(var, 0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context).flatten()
        reward = float(reward)
        
        # Update A and b with discounting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        self.t += 1
# EVOLVE-BLOCK-END
