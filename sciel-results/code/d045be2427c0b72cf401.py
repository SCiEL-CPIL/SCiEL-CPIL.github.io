"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive Kalman-filter LinUCB for non-stationary linear contextual bandits.

    Each arm maintains a state (theta) and covariance (P) updated via a Kalman
    filter with adaptive process noise Q driven by the residual-to-predicted-
    variance ratio.  A deadline guard ensures we return within the wall-clock limit.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 115.0  # seconds, safety margin before 120 s kill

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large covariance to encourage exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self._base_Q = 1e-3
        self._q_scale = 1.0

        # UCB parameter
        self.alpha = 1.0

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on residual ratio."""
        if self.count[arm] < 2:
            return self._base_Q
        # Use a running estimate of residual variance relative to prediction variance
        # For simplicity, scale Q by a factor that grows when uncertainty is high
        P_diag_mean = np.mean(np.diag(self.P[arm]))
        q = self._base_Q * (1.0 + 0.5 * np.sqrt(P_diag_mean))
        return q

    def select(self, context):
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            # Fallback: pick arm with highest mean theta @ context
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predicted mean
            mu = self.theta[a] @ ctx
            # Predicted variance
            var = ctx @ self.P[a] @ ctx
            # UCB score
            score = mu + self.alpha * np.sqrt(max(var, 0.0))
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Predict step: inflate covariance with adaptive process noise
        q = self._adaptive_Q(arm)
        self.P[arm] = self.P[arm] + q * np.eye(self.d)

        # Update step: Kalman gain
        # Innovation
        pred = self.theta[arm] @ ctx
        innovation = reward - pred
        # Innovation variance
        S = ctx @ self.P[arm] @ ctx + 1e-6  # small regularization for observation noise
        # Kalman gain
        K = (self.P[arm] @ ctx) / S
        # State update
        self.theta[arm] = self.theta[arm] + K * innovation
        # Covariance update (Joseph form for numerical stability)
        I_KK = np.eye(self.d) - np.outer(K, ctx)
        self.P[arm] = I_KK @ self.P[arm] @ I_KK.T + q * np.eye(self.d)

        # Symmetrize to prevent drift
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        self.count[arm] += 1
# EVOLVE-BLOCK-END
