# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Dual-timescale Kalman filter for non-stationary linear contextual bandits.

    Models each arm's weight vector as a random walk with adaptive process noise.
    Fast timescale: reward estimation via Kalman update.
    Slow timescale: drift tracking via adaptive process noise inflation.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 10s safety margin before 150s

        # Per-arm state: theta (mean), P (covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large covariance to encourage exploration
        self.P = [np.eye(dim) * 50.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self.base_Q = 0.01  # base process noise scalar
        self.obs_noise = 1.0  # observation noise estimate
        self.error_history = [np.zeros(1) for _ in range(n_arms)]  # rolling prediction errors
        self.error_window = 50

        # Exploration coefficient
        self.alpha = 0.5

        # Track best arm for deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def _adaptive_Q(self, arm, pred_error):
        """Compute adaptive process noise covariance for a given arm.
        
        Scales base process noise by ratio of prediction error to observation noise.
        Uses a rolling average of recent errors for stability.
        """
        # Update error history
        self.error_history[arm] = np.append(self.error_history[arm], pred_error)
        if len(self.error_history[arm]) > self.error_window:
            self.error_history[arm] = self.error_history[arm][-self.error_window:]
        
        # Compute rolling mean of squared errors
        mse = np.mean(self.error_history[arm] ** 2)
        # Scale process noise: higher prediction error -> more process noise
        scale = 1.0 + mse / (self.obs_noise + 1e-8)
        # Cap the scale to avoid instability
        scale = min(scale, 10.0)
        return self.base_Q * scale * np.eye(self.d)

    def select(self, context):
        # Deadline guard: if we're close to the limit, return best known arm
        if time.time() > self.deadline:
            return self.best_arm
        
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Predicted reward
            mu = self.theta[a] @ context
            # Uncertainty (posterior std of predicted reward)
            var = context @ self.P[a] @ context
            std = np.sqrt(max(var, 0.0))
            # UCB-style score with exploration bonus
            scores[a] = mu + self.alpha * std
        
        # Select best arm
        arm = int(np.argmax(scores))
        self.best_arm = arm
        return arm

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Prediction error
        pred = self.theta[arm] @ context
        error = reward - pred
        
        # Track best reward for deadline fallback
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = arm

        # Adaptive process noise (predict step)
        Q = self._adaptive_Q(arm, error)
        P_pred = self.P[arm] + Q

        # Kalman update (fast timescale)
        # K = P_pred @ x^T / (x^T @ P_pred @ x + R)
        Px = P_pred @ context
        denom = context @ Px + self.obs_noise
        K = Px / max(denom, 1e-12)
        
        # Update mean and covariance
        self.theta[arm] = self.theta[arm] + K * error
        # Joseph form for numerical stability: P = (I - K x^T) P_pred (I - K x^T)^T + K R K^T
        IK = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = IK @ P_pred @ IK.T + np.outer(K, K) * self.obs_noise
        
        # Ensure symmetry
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)
        
        self.count[arm] += 1
# EVOLVE-BLOCK-END
