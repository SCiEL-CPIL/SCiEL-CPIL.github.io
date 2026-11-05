"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Policy:
    """Exponential Weighted Linear Bandit with Forgetting Factor."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Forgetting factor, effective window ~20
        self.beta = 2.0    # Exploration parameter
        self.lambda_reg = 1.0 # Initial ridge regularization
        # Initialize A and b for each arm
        # A is dxd, b is d
        self.A_list = [self.lambda_reg * np.eye(self.d) for _ in range(self.n)]
        self.b_list = [np.zeros(self.d) for _ in range(self.n)]
        self.t = 0

    def select(self, context):
        context = np.asarray(context).flatten()
        scores = []
        for a in range(self.n):
            A = self.A_list[a]
            b = self.b_list[a]
            
            try:
                # Solve A z = context for z = A_inv @ context
                z = np.linalg.solve(A, context)
                var = context @ z
                # Solve A theta = b for theta
                theta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback to prior if singular
                z = context / self.lambda_reg
                var = context @ z
                theta = b / self.lambda_reg
            
            pred = theta @ context
            score = pred + self.beta * np.sqrt(max(var, 0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context).flatten()
        reward = float(reward)
        
        # Recursive update with forgetting factor
        self.A_list[arm] = self.gamma * self.A_list[arm] + np.outer(context, context)
        self.b_list[arm] = self.gamma * self.b_list[arm] + reward * context
        self.t += 1
# EVOLVE-BLOCK-END
