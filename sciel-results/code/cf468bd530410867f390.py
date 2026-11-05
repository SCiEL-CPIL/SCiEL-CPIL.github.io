"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Policy:
    """Discounted Linear Bandit with Adaptive Exploration."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.alpha = 0.95  # Forgetting factor
        self.lambda_reg = 1.0  # Initial regularization
        self.beta_start = 5.0  # Initial exploration
        self.beta_min = 0.5    # Minimum exploration
        self.beta_decay = 0.999 # Decay rate for beta
        
        # State for each arm: A (covariance), b (cross-product)
        # Initialize A as lambda * I
        self.A = [self.lambda_reg * np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        self.t = 0

    def select(self, context):
        context = np.asarray(context).flatten()
        scores = []
        current_beta = self.beta_start * (self.beta_decay ** self.t)
        if current_beta < self.beta_min:
            current_beta = self.beta_min

        for a in range(self.n):
            # Compute theta and variance
            # theta = A^-1 b
            # var = x^T A^-1 x
            try:
                A_inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                # Fallback if singular (should not happen with lambda_reg)
                A_inv = np.eye(self.d) / self.lambda_reg
            
            theta = A_inv @ self.b[a]
            pred = theta @ context
            var = context @ A_inv @ context
            score = pred + current_beta * np.sqrt(max(var, 0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context).flatten()
        reward = float(reward)
        
        # Update statistics with forgetting
        # A = alpha * A + x x^T
        # b = alpha * b + r x
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
        
        self.t += 1
# EVOLVE-BLOCK-END
