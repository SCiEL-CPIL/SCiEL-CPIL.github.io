"""Bootstrap spawn: direct approach for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm ridge regression with exponential forgetting (discounted least squares)
    - Change Point Detection with model reset
    - UCB-style exploration
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s kill

        # Per-arm parameters: V is the posterior covariance.
        # Initialize with high uncertainty (large V) to encourage exploration.
        self.V = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # For change point detection
        self.cum_err = [0.0] * n_arms
        self.err_threshold = 50.0  # Threshold for resetting
        self.err_decay = 0.99      # Decay for cumulative error to allow recovery

        # Exploration
        self.C = 1.5
        self.base_gamma = 0.97

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_C(self, arm):
        """
        Standard UCB exploration.
        """
        c = self.counts[arm]
        if c < 5:
            return self.C * 2.0
        return self.C

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        # UCB with adaptive exploration
        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(context @ self.V[a] @ context)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error before updating
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = reward - pred
        err_sq = err * err
        
        # Change Point Detection Logic
        # Decay the cumulative error to allow the system to "forget" old errors
        self.cum_err[arm] = self.cum_err[arm] * self.err_decay + err_sq
        
        if self.cum_err[arm] > self.err_threshold:
            # Reset the model for this arm
            self.V[arm] = np.eye(self.d) * 10.0
            self.b[arm] = np.zeros(self.d)
            self.cum_err[arm] = 0.0

        # Adaptive forgetting
        gamma = self.base_gamma

        # Update inverse covariance using Sherman-Morrison with forgetting
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V[arm] = np.eye(self.d) * 10.0

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
