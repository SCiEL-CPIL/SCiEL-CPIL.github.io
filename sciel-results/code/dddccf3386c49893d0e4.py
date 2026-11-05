"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Lightweight Non-Stationary Linear Bandit Policy.
    
    Strategy:
    - Per-arm weight vectors updated via Stochastic Gradient Descent (SGD) step.
    - Epsilon-Greedy exploration.
    - No matrix inversions or covariance tracking.
    - Deadline guard to prevent timeouts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize weight vectors to zero
        self.thetas = [np.zeros(dim) for _ in range(n_arms)]
        
        # Hyperparameters
        # Learning rate for the gradient step. 
        # In non-stationary settings, a moderate LR helps track changes.
        self.alpha = 0.1 
        # Epsilon for exploration. 
        # Fixed epsilon is simpler and often robust. 
        # 0.1-0.2 is a common range for 5 arms.
        self.epsilon = 0.15
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 10s safety margin

    def _check_deadline(self):
        """Check if we are approaching the time limit."""
        return time.time() > self.deadline

    def select(self, context):
        # Deadline check: if we are running out of time, just return a random arm 
        # or the best known arm quickly to avoid hanging in complex logic.
        # In this simple policy, select is already O(n*d), so it's very fast.
        if self._check_deadline():
            return int(np.random.randint(self.n))

        # Epsilon-Greedy
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n))
        
        # Greedy: compute predicted reward for each arm
        # context is a vector of shape (d,)
        # thetas[a] is a vector of shape (d,)
        rewards = np.array([self.thetas[a] @ context for a in range(self.n)])
        return int(np.argmax(rewards))

    def update(self, context, arm, reward):
        # Deadline check
        if self._check_deadline():
            return

        # Stochastic Gradient Descent step for linear regression
        # Prediction: pred = theta[arm] @ context
        # Error: e = reward - pred
        # Gradient of (reward - theta @ context)^2 w.r.t theta is -2 * e * context
        # Update: theta += alpha * 2 * e * context
        # We absorb the factor of 2 into alpha for simplicity, or just use alpha directly
        # as a learning rate for the least squares objective.
        
        pred = self.thetas[arm] @ context
        error = reward - pred
        
        # Update the weight vector for the pulled arm
        self.thetas[arm] += self.alpha * error * context
# EVOLVE-BLOCK-END
