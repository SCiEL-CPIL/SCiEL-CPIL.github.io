"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-Stationary Linear Contextual Bandit Policy.
    
    Strategy:
    - Per-arm weight vectors updated via SGD with a fixed learning rate.
    - Epsilon-Greedy exploration with a fixed epsilon.
    - Deadline guard to prevent timeouts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize weight vectors to zero
        self.thetas = [np.zeros(dim) for _ in range(n_arms)]
        
        # Hyperparameters
        self.alpha = 0.1 
        self.epsilon = 0.15
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 10s safety margin

    def _check_deadline(self):
        """Check if we are approaching the time limit."""
        return time.time() > self.deadline

    def select(self, context):
        if self._check_deadline():
            return int(np.random.randint(self.n))

        # Epsilon-Greedy
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n))
        
        # Greedy: compute predicted reward for each arm
        rewards = np.array([self.thetas[a] @ context for a in range(self.n)])
        return int(np.argmax(rewards))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return

        pred = self.thetas[arm] @ context
        error = reward - pred
        
        self.thetas[arm] += self.alpha * error * context
# EVOLVE-BLOCK-END
