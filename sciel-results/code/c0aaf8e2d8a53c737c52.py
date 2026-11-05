# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Dual-timescale Kalman filter for non-stationary linear contextual bandits.

    Models each arm's weight vector as a random walk with adaptive process noise.
    Uses a fast timescale for reward estimation and a slow timescale for drift tracking.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]  # covariance, start large
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self.Q_base = 0.01  # base process noise variance
        self.Q_max = 1.0
        self.Q_min = 1e-4
        self.obs_noise = 1.0  # assumed observation noise variance
        self.error_history = [np.zeros(50) for _ in range(n_arms)]
        self.error_ptr = [0 for _ in range(n_arms)]

        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # safety margin before 150s kill

    def _adaptive_Q(self, arm):
        """Adaptively scale process noise based on recent prediction errors."""
        if self.count[arm] < 5:
            return self.Q_base
        hist = self.error_history[arm]
        n_err = min(self.count[arm], len(hist))
        if n_err == 0:
            return self.Q_base
        recent_errors = hist[:n_err] if n_err < len(hist) else hist
        mean_sq_err = np.mean(recent_errors ** 2)
        ratio = mean_sq_err / (self.obs_noise + 1e-8)
        Q = self.Q_base * (1.0 + ratio)
        return np.clip(Q, self.Q_min, self.Q_max)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline check: if close to deadline, exploit best estimate
        if time.time() > self.deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context
            # Uncertainty: sqrt(x^T P x)
            P_ctx = self.P[a] @ context
            variance = context @ P_ctx
            if variance < 0:
                variance = 0.0
            std = np.sqrt(max(variance, 0.0))
            # UCB with adaptive exploration coefficient
            beta = 1.0 + 0.5 * np.log(1.0 + self.t)
            scores.append(mean + beta * std)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Predict step: inflate covariance by process noise
        Q = self._adaptive_Q(arm)
        self.P[arm] = self.P[arm] + Q * np.eye(self.d)

        # Update step: Kalman gain
        P_ctx = self.P[arm] @ context
        denom = context @ P_ctx + self.obs_noise
        if denom < 1e-12:
            denom = 1e-12
        K = P_ctx / denom  # Kalman gain vector

        # Prediction error
        predicted_reward = self.theta[arm] @ context
        innovation = reward - predicted_reward

        # Track error for adaptive Q
        self.error_history[arm][self.error_ptr[arm]] = innovation
        self.error_ptr[arm] = (self.error_ptr[arm] + 1) % len(self.error_history[arm])

        # Update state
        self.theta[arm] = self.theta[arm] + K * innovation

        # Update covariance: (I - K x^T) P
        KxT = np.outer(K, context)
        self.P[arm] = (np.eye(self.d) - KxT) @ self.P[arm]

        # Symmetrize to avoid numerical drift
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        self.count[arm] += 1
# EVOLVE-BLOCK-END
