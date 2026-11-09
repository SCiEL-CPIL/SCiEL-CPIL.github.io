"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-parametric sliding-window kernel policy for non-stationary linear contextual bandits.
    Instead of maintaining a global linear weight vector per arm, we maintain a buffer of recent
    (context, reward) pairs and compute arm values via local ridge regression on this window.
    This allows the policy to adapt quickly to changes in the underlying reward function without
    relying on a fixed linear structure that may become stale.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.buffer = [[] for _ in range(n_arms)]
        self.window_size = 50  # Sliding window of recent observations per arm
        self.reg = 1.0  # Ridge regression regularization
        self.exploration_weight = 1.0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin

    def _local_value(self, arm, context):
        """
        Estimate the value of 'arm' for 'context' using local ridge regression
        on the recent buffer for that arm.
        """
        buf = self.buffer[arm]
        if len(buf) == 0:
            return 0.0, 0.0  # Mean, Uncertainty

        # Use only the last 'window_size' observations
        recent = buf[-self.window_size:]
        X = np.array([x for x, r in recent])
        y = np.array([r for x, r in recent])

        # Local ridge regression: theta = (X^T X + reg * I)^-1 X^T y
        # To improve locality, we could weight recent points more, but let's start with uniform in window
        # Actually, a weighted kernel would be better. Let's use exponential decay weights within the window.
        n_pts = len(recent)
        if n_pts == 1:
            return y[0], 0.0

        # Weights: exponential decay, most recent = 1.0
        weights = np.exp(-0.1 * np.arange(n_pts - 1, -1, -1))  # Most recent (last) has weight 1
        weights /= np.sum(weights)

        # Weighted ridge regression
        W = np.diag(weights)
        XtW = X.T @ W
        XtWX = XtW @ X + self.reg * np.eye(self.d)
        XtWy = XtW @ y

        try:
            theta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d)

        mean_val = theta @ context

        # Uncertainty: approximate via diagonal of inverse
        # Var = (X^T W X + reg I)^-1
        try:
            inv_cov = np.linalg.inv(XtWX)
            var = context @ inv_cov @ context
            uncertainty = np.sqrt(var)
        except np.linalg.LinAlgError:
            uncertainty = 0.0

        return mean_val, uncertainty

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: random or most recently best
            return int(np.random.randint(self.n))

        scores = []
        for a in range(self.n):
            mean_val, uncertainty = self._local_value(a, context)
            # UCB-style score
            score = mean_val + self.exploration_weight * uncertainty
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.buffer[arm].append((context, reward))
        # Trim buffer to window size
        if len(self.buffer[arm]) > self.window_size:
            self.buffer[arm].pop(0)

# EVOLVE-BLOCK-END
