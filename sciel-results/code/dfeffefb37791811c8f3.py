# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary linear bandit using Kalman filter with innovation-gated process noise."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim + 1  # Augmented dimension with intercept
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

        # Kalman filter state per arm: theta (mean), P (covariance)
        self.theta = [np.zeros(self.d) for _ in range(n_arms)]
        # Initialize covariance large to encourage exploration
        self.P = [np.eye(self.d) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        self.alpha = 1.0  # Exploration parameter for UCB
        self.total_rounds = 0
        self.reward_history = [0.0 for _ in range(n_arms)]
        self.residual_history = [0.0 for _ in range(n_arms)]
        # Process noise Q per arm (random walk on weights)
        self.Q = [np.eye(self.d) * 0.01 for _ in range(n_arms)]
        # Observation noise
        self.R = 0.1
        # Innovation gating parameters
        self.innovation_threshold = 1.5  # z-score threshold for detecting regime shift
        self.q_base = 0.005
        self.q_max = 0.5

    def _augment(self, context):
        """Augment context with intercept term."""
        context = np.asarray(context, dtype=np.float64)
        return np.append(context, 1.0)

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        return time.time() > self.deadline

    def _select_safe_arm(self):
        """Return a safe arm when deadline is approaching."""
        if any(r != 0 for r in self.reward_history):
            return int(np.argmax(self.reward_history))
        return 0

    def select(self, context):
        if self._check_deadline():
            return self._select_safe_arm()

        x = self._augment(context)
        scores = np.zeros(self.n)

        for a in range(self.n):
            # Predicted reward
            pred = self.theta[a] @ x

            # Uncertainty: sqrt(x^T P x)
            uncertainty = np.sqrt(max(1e-10, x @ self.P[a] @ x))

            scores[a] = pred + self.alpha * uncertainty

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return

        x = self._augment(context)
        self.total_rounds += 1

        # Update reward history
        self.reward_history[arm] = 0.95 * self.reward_history[arm] + 0.05 * reward

        # --- Kalman Predict step ---
        # theta_pred = theta (random walk)
        # P_pred = P + Q
        self.P[arm] = self.P[arm] + self.Q[arm]

        # --- Kalman Update step ---
        # Innovation
        y = reward - x @ self.theta[arm]

        # Innovation covariance
        S = x @ self.P[arm] @ x + self.R
        if S < 1e-10:
            S = 1e-10

        # Kalman gain
        K = self.P[arm] @ x / S

        # Update theta
        self.theta[arm] = self.theta[arm] + K * y

        # Update covariance (Joseph form for numerical stability)
        I_Kx = np.eye(self.d) - np.outer(K, x)
        self.P[arm] = I_Kx @ self.P[arm] @ I_Kx.T + K * self.R * K.T

        # Ensure symmetry
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        self.count[arm] += 1

        # Track residual for adaptive process noise
        self.residual_history[arm] = 0.9 * self.residual_history[arm] + 0.1 * (y * y)

        # Innovation-gated process noise adaptation
        if self.total_rounds > 5:
            # Compute standardized innovation (z-score)
            innovation_std = np.sqrt(S)
            if innovation_std > 1e-10:
                z_score = abs(y) / innovation_std
            else:
                z_score = 0.0

            # If innovation is large, likely a regime shift -> increase Q significantly
            if z_score > self.innovation_threshold:
                # Strong adaptation: jump Q up quickly
                new_q = min(self.q_max, self.Q[arm].trace() / self.d * 3.0)
                self.Q[arm] = np.eye(self.d) * new_q
            else:
                # Normal operation: slowly adapt Q based on residual
                mean_residual = np.mean([self.residual_history[a] for a in range(self.n)])
                adaptive_q = self.q_base * (1.0 + 2.0 * min(mean_residual, 2.0))
                target_Q = np.eye(self.d) * adaptive_q
                self.Q[arm] = 0.9 * self.Q[arm] + 0.1 * target_Q

        # Adaptive exploration: reduce alpha over time
        self.alpha = max(0.3, 1.0 * np.exp(-self.total_rounds / 200.0))
# EVOLVE-BLOCK-END
