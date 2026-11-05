"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Policy:
    """Discounted Linear Bandit for Non-Stationary Settings."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95       # Discount factor: closer to 1 means longer memory
        self.beta = 3.0         # Exploration parameter (tuned for discounted variance)
        self.lambda_reg = 1.0   # Ridge regularization for stability
        
        # Initialize sufficient statistics for each arm
        # A: covariance matrix, b: cross-covariance vector
        self.A = [self.lambda_reg * np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        self.A_inv = [np.linalg.inv(self.A[i]) for i in range(self.n)]
        self.t = 0

    def select(self, context):
        context = np.asarray(context).flatten()
        scores = []
        for a in range(self.n):
            # Predicted mean
            theta = self.A_inv[a] @ self.b[a]
            pred = theta @ context
            
            # Variance term for UCB
            var = context @ self.A_inv[a] @ context
            
            # UCB score
            score = pred + self.beta * np.sqrt(max(var, 0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context).flatten()
        reward = float(reward)
        
        # Discount previous statistics
        self.A[arm] *= self.gamma
        self.b[arm] *= self.gamma
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Update inverse using Sherman-Morrison formula for efficiency
        # A_new = gamma * A_old + x x^T
        # We maintain A_inv. 
        # Let A_t = gamma A_{t-1} + x x^T
        # A_t^{-1} = (gamma A_{t-1} + x x^T)^{-1}
        # Using Woodbury identity or direct inversion if d is small.
        # Since d is likely small (e.g., 5-10), direct inversion is safe and fast enough.
        # However, to be robust, we use np.linalg.inv.
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            self.A_inv[arm] = np.eye(self.d) / self.lambda_reg

        self.t += 1
# EVOLVE-BLOCK-END
