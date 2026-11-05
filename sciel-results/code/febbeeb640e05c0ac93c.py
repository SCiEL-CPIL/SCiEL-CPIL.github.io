# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Dual-timescale Kalman filter for non-stationary linear contextual bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 115.0  # Safety margin before 120s limit

        # Kalman filter state per arm
        # theta: current estimate of weight vector
        # P: covariance of theta estimate
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]  # Large initial uncertainty

        # Process noise parameters
        self.Q_base = 0.01  # Base process noise covariance scale
        self.sigma2 = 1.0   # Observation noise variance
        self.alpha = 0.1    # Adaptation rate for process noise scaling

        # Adaptive process noise scaling per arm
        self.pred_error = [0.0 for _ in range(n_arms)]
        self.Q_scale = [1.0 for _ in range(n_arms)]

        # UCB exploration coefficient
        self.beta = 1.5

        # Visit counts for calibration
        self.visits = [0 for _ in range(n_arms)]

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Return the arm with highest mean estimate (exploit)
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64).flatten()
        scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context
            # UCB bonus based on Kalman filter covariance
            var = context @ self.P[a] @ context
            bonus = self.beta * np.sqrt(var)
            scores.append(mean + bonus)

        # Thompson sampling alternative: sample from posterior
        # Using UCB for stability
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        self.visits[arm] += 1

        # Prediction step: propagate uncertainty
        # theta_pred = theta (random walk with zero mean)
        # P_pred = P + Q * Q_scale
        Q = self.Q_base * self.Q_scale[arm] * np.eye(self.d)
        P_pred = self.P[arm] + Q

        # Update step: Kalman gain
        # K = P_pred @ x / (x^T P_pred x + sigma2)
        Px = P_pred @ context
        denom = context @ Px + self.sigma2
        K = Px / denom  # shape: (d,)

        # Innovation (prediction error)
        y_pred = self.theta[arm] @ context
        innovation = reward - y_pred

        # Update theta and P
        self.theta[arm] = self.theta[arm] + K * innovation
        # Joseph form for numerical stability
        I_Kx = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_Kx @ P_pred @ I_Kx.T + self.sigma2 * np.outer(K, K)

        # Adapt process noise scaling based on prediction error
        # If innovation is large relative to expected, increase Q_scale
        expected_innovation_std = np.sqrt(denom)
        error_ratio = (innovation ** 2) / (self.sigma2 + 1e-6)
        # Exponential moving average of error ratio
        self.pred_error[arm] = (1 - self.alpha) * self.pred_error[arm] + self.alpha * error_ratio
        # Scale Q: higher when error is high
        self.Q_scale[arm] = 1.0 + self.pred_error[arm]
        # Cap to prevent instability
        self.Q_scale[arm] = min(self.Q_scale[arm], 100.0)

        self.t += 1
# EVOLVE-BLOCK-END
