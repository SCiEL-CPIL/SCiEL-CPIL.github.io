# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary linear bandit using per-arm Kalman filter with adaptive process noise
    and Thompson Sampling exploration."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # 5s safety margin

        # Kalman filter state per arm:
        #   theta: mean of weight vector (dim,)
        #   P:     covariance of weight vector (dim, dim)
        #   Q:     process noise covariance (dim, dim) — adaptive
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Large initial covariance = high uncertainty
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        # Initial process noise: small but non-zero
        self.Q = [np.eye(dim) * 0.01 for _ in range(n_arms)]

        # For adaptive Q estimation: track recent prediction errors per arm
        self.pred_err_sq = [0.0 for _ in range(n_arms)]
        self.err_count = [0 for _ in range(n_arms)]
        self.total_rounds = 0

        # Measurement noise variance
        self.R = 1.0

        # Thompson sampling: we sample from N(theta, P)
        # Precompute Cholesky factors updated each round
        self.cholesky = [None for _ in range(n_arms)]
        self._update_cholesky()

    def _update_cholesky(self):
        for a in range(self.n):
            try:
                self.cholesky[a] = np.linalg.cholesky(self.P[a] + 1e-8 * np.eye(self.d))
            except np.linalg.LinAlgError:
                self.cholesky[a] = np.eye(self.d) * 1e-4

    def _check_deadline(self):
        return time.time() >= self.deadline

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # If near deadline, just pick the arm with highest predicted reward
        if self._check_deadline():
            preds = np.array([self.theta[a] @ context for a in range(self.n)])
            return int(np.argmax(preds))

        # Thompson Sampling: draw one sample per arm from N(theta_a, P_a)
        # score_a = (theta_a + eps_a) @ context where eps_a ~ N(0, P_a)
        scores = np.zeros(self.n)
        for a in range(self.n):
            if self.cholesky[a] is not None:
                eps = self.cholesky[a] @ np.random.randn(self.d)
            else:
                eps = np.zeros(self.d)
            scores[a] = (self.theta[a] + eps) @ context

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.total_rounds += 1

        # --- Adaptive process noise Q estimation ---
        # Prediction: y_pred = theta[arm] @ context
        # Innovation (prediction error): e = reward - y_pred
        y_pred = self.theta[arm] @ context
        innovation = reward - y_pred
        self.pred_err_sq[arm] = 0.9 * self.pred_err_sq[arm] + 0.1 * (innovation ** 2)
        self.err_count[arm] += 1

        # Estimate process noise variance from innovation variance
        # If innovations are large, the weights are drifting → increase Q
        if self.err_count[arm] > 5:
            innov_var = self.pred_err_sq[arm]
            # Scale Q: base 0.01, increase with innovation variance
            # Use sqrt to avoid over-amplification
            q_scale = 0.01 + 0.1 * np.sqrt(innov_var)
            q_scale = np.clip(q_scale, 0.005, 1.0)
            self.Q[arm] = np.eye(self.d) * q_scale

        # --- Kalman Filter Update ---
        # H = context (1 x d), measurement z = reward (scalar)
        # P_pred = P + Q  (prediction step)
        P_pred = self.P[arm] + self.Q[arm]

        # S = H P_pred H^T + R  (innovation covariance)
        H = context  # shape (d,)
        S = float(H @ P_pred @ H) + self.R

        # K = P_pred H^T / S  (Kalman gain, shape (d,))
        K = (P_pred @ H) / S

        # theta = theta + K * innovation
        self.theta[arm] = self.theta[arm] + K * innovation

        # P = (I - K H) P_pred  (covariance update)
        I_KH = np.eye(self.d) - np.outer(K, H)
        self.P[arm] = I_KH @ P_pred

        # Ensure symmetry
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        # Update Cholesky for next Thompson Sampling draw
        try:
            self.cholesky[arm] = np.linalg.cholesky(self.P[arm] + 1e-8 * np.eye(self.d))
        except np.linalg.LinAlgError:
            self.cholesky[arm] = np.eye(self.d) * 1e-4
# EVOLVE-BLOCK-END
