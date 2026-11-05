# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive Kalman-filter LinUCB for non-stationary linear contextual bandits.

    Models each arm's weight vector as a random walk with adaptive process noise.
    Uses residual-driven Q adaptation to track drift smoothly.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = 120.0  # safety margin before 150s kill

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]  # covariance
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise (per-arm)
        self.Q = [0.01 * np.eye(dim) for _ in range(n_arms)]
        self.residual_hist = [[] for _ in range(n_arms)]

        # Exploration constant
        self.alpha = 1.0

        # For fast fallback if computation is too slow
        self._use_fast = False

    def select(self, context):
        # Deadline guard
        if time.time() - self._t0 > self._deadline:
            # Return best arm based on current theta estimates
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predict step: inflate covariance with process noise
            P_pred = self.P[a] + self.Q[a]

            # Compute mean and variance of reward estimate
            mean = self.theta[a] @ context
            var = context @ P_pred @ context
            ucb = mean + self.alpha * np.sqrt(max(var, 0.0))
            scores.append(ucb)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        a = arm

        # Predict step
        P_pred = self.P[a] + self.Q[a]

        # Innovation (residual)
        y = reward - self.theta[a] @ context
        S = context @ P_pred @ context + 1.0  # observation noise = 1

        # Kalman gain
        K = P_pred @ context / S

        # Update state
        self.theta[a] = self.theta[a] + K * y
        self.P[a] = P_pred - np.outer(K, context @ P_pred)

        # Ensure symmetry and positive semi-definiteness
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        # Clip small negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(self.P[a])
        eigvals = np.maximum(eigvals, 1e-8)
        self.P[a] = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self.count[a] += 1

        # Adaptive process noise: track residual magnitude
        self.residual_hist[a].append(abs(y))
        if len(self.residual_hist[a]) > 50:
            self.residual_hist[a].pop(0)

        # Update Q based on residual statistics
        if self.count[a] >= 5:
            mean_res = np.mean(self.residual_hist[a])
            # Scale Q to match observed drift level
            q_scale = max(0.001, min(0.5, mean_res * 0.1))
            self.Q[a] = q_scale * np.eye(self.d)
# EVOLVE-BLOCK-END
