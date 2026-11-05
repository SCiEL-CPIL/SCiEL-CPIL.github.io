# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Global linear bandit with arm-specific biases via augmented context.
    Uses a single shared weight vector with one-hot arm indicators to capture
    arm-specific biases, improving sample efficiency and numerical stability.
    Adaptive forgetting factor tracks non-stationary changes.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Augmented dimension: original features + one-hot arm indicators
        self.d_aug = dim + n_arms
        # Single global model
        self.A = np.eye(self.d_aug)  # Inverse covariance (regularized)
        self.b = np.zeros(self.d_aug)
        # Adaptive forgetting parameters
        self.base_gamma = 0.98
        self.gamma_min = 0.90
        self.gamma_max = 0.999
        self.gamma = self.base_gamma
        # Ridge for numerical stability
        self.ridge = 1e-3
        # Exploration parameter
        self.beta0 = 1.0
        self.beta_schedule = 0.999
        self.beta = self.beta0
        # Track rounds for potential adaptive exploration
        self.t = 0
        # Deadline guard
        self.t0 = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

    def _augment(self, context, arm):
        """Augment context with one-hot arm indicator."""
        x = np.zeros(self.d_aug)
        x[:self.d] = context
        x[self.d + arm] = 1.0
        return x

    def select(self, context):
        # Deadline guard
        if time.time() - self.t0 > self.deadline:
            return int(np.random.randint(self.n))

        scores = []
        # Precompute inverse once
        inv_A = np.linalg.inv(self.A)
        theta = inv_A @ self.b

        for a in range(self.n):
            x_aug = self._augment(context, a)
            # Predicted mean
            mu = theta @ x_aug
            # Uncertainty bonus
            var = x_aug @ inv_A @ x_aug
            bonus = self.beta * np.sqrt(max(var, 0.0))
            scores.append(mu + bonus)

        scores = np.array(scores)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1

        # Adaptive forgetting: adjust gamma based on recent performance
        # Use a simple heuristic: if we're in a new "regime", forget faster
        # Here we use a slow decay toward base_gamma with occasional resets
        # based on round number
        if self.t % 50 == 0:
            # Periodically reset gamma to base to avoid over-forgetting
            self.gamma = self.base_gamma
        else:
            # Slight adaptation: reduce gamma slightly each round to track changes
            self.gamma = max(self.gamma_min, self.gamma * 0.9999)

        # Update the global model with augmented context
        x_aug = self._augment(context, arm)

        # Apply forgetting factor
        self.A = self.gamma * self.A + np.outer(x_aug, x_aug)
        self.b = self.gamma * self.b + reward * x_aug

        # Add ridge for numerical stability
        self.A += self.ridge * np.eye(self.d_aug)

        # Adaptive exploration: decrease beta over time but with a floor
        self.beta = max(0.5, self.beta * self.beta_schedule)
# EVOLVE-BLOCK-END
