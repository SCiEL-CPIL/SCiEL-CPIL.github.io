# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Global linear contextual bandit with one-hot arm indicators.
    Context is augmented with one-hot arm encodings so a single shared
    weight vector captures both context effects and arm-specific biases.
    Uses exponentially-weighted recursive least squares with adaptive
    forgetting and UCB exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms  # augmented dimension

        # Adaptive forgetting parameters
        self.gamma_base = 0.95
        self.gamma_min = 0.80
        self.gamma_max = 0.99

        # Exploration
        self.beta = 1.5

        # Ridge for numerical stability
        self.ridge = 1e-3

        # Deadline guard
        self._t0 = time.time()
        self._deadline = 145.0  # safety margin before 150s kill

        # Global model state: theta (d_aug,), P (d_aug x d_aug)
        self.theta = np.zeros(self.d_aug)
        self.P = np.eye(self.d_aug) * (1.0 / self.ridge)

        # For adaptive forgetting: track recent prediction errors
        self.recent_errors = []
        self.error_window = 20

        # One-hot cache
        self._onehot = [np.zeros(self.d_aug) for _ in range(n_arms)]
        for a in range(n_arms):
            self._onehot[a][self.d + a] = 1.0

    def _augment(self, context, arm):
        """Augment context with one-hot arm indicator."""
        x = np.zeros(self.d_aug)
        x[:self.d] = context
        x[self.d + arm] = 1.0
        return x

    def _adaptive_gamma(self):
        """Adapt forgetting rate based on recent prediction error."""
        if len(self.recent_errors) < 5:
            return self.gamma_base

        recent = self.recent_errors[-self.error_window:]
        mean_err = np.mean(np.abs(recent))

        # Higher error -> more aggressive forgetting (lower gamma)
        # Scale: if mean_err > 1, push gamma toward gamma_min
        # If mean_err < 0.1, push gamma toward gamma_max
        if mean_err > 2.0:
            return self.gamma_min
        elif mean_err < 0.2:
            return self.gamma_max
        else:
            # Linear interpolation
            t = np.clip((mean_err - 0.2) / (2.0 - 0.2), 0, 1)
            return self.gamma_max - t * (self.gamma_max - self.gamma_min)

    def select(self, context):
        if time.time() - self._t0 > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            x = self._augment(context, a)
            mu = self.theta @ x
            var = x @ self.P @ x
            ucb = mu + self.beta * np.sqrt(max(var, 0.0))
            scores.append(ucb)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        x = self._augment(context, arm)

        # Predicted value
        mu = self.theta @ x
        error = reward - mu

        # Track error for adaptive forgetting
        self.recent_errors.append(abs(error))
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)

        # Adaptive forgetting factor
        gamma = self._adaptive_gamma()

        # Recursive least squares update with forgetting
        # P = (1/gamma) * (P - P x (x^T P x + 1/gamma)^{-1} x^T P)
        # theta = theta + P x * error

        Px = self.P @ x
        denom = 1.0 + x @ Px

        # Update P with forgetting
        self.P = (self.P - np.outer(Px, Px) / denom) / gamma

        # Add ridge for stability
        self.P += self.ridge * np.eye(self.d_aug)

        # Update theta
        self.theta = self.theta + Px * error
    # EVOLVE-BLOCK-END
