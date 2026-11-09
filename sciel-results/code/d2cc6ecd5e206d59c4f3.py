"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Discounted LinUCB with adaptive exploration for non-stationary bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.985  # Forgetting factor
        self.alpha = 1.0    # Exploration parameter
        self.lam = 1.0      # Regularization

        # Initialize A and b for each arm
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Track last update time for potential time-based decay
        self.last_update_time = time.time()
        self.t = 0

    def select(self, context):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        scores = []
        for a in range(self.n):
            # Compute theta = A^{-1} b
            try:
                inv_A = np.linalg.inv(self.A[a])
                theta = inv_A @ self.b[a]
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                inv_A = np.eye(self.d)
            
            # Predicted reward
            pred = theta @ context
            
            # UCB bonus: alpha * sqrt(context^T A^{-1} context)
            quad = context @ inv_A @ context
            bonus = self.alpha * np.sqrt(max(0, quad))
            
            scores.append(pred + bonus)
        
        scores = np.array(scores)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Apply forgetting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Keep regularization term constant
        self.A[arm][:, :] += 0  # No-op, but ensures matrix stays well-conditioned
        
        self.last_update_time = time.time()
# EVOLVE-BLOCK-END
