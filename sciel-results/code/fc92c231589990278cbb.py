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

        # Adaptive process noise per arm
        self.Q = [1e-4 * np.eye(dim) for _ in range(n_arms)]

        # Prediction error tracking for adaptive Q
        self.pred_errors = [0.0 for _ in range(n_arms)]
        self.error_count = [0 for _ in range(n_arms)]

        # Deadline guard
        self.start_time = time.time()
        self.time_limit = 140.0  # safety margin before 150s kill

        # Exploration bonus base
        self.beta = 1.5

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on recent prediction errors."""
        if self.error_count[arm] < 5:
            return self.Q[arm]
        
        # Scale process noise by recent prediction error magnitude
        recent_err = self.pred_errors[arm]
        # Use a smooth scaling: base + adaptive component
        base = 1e-4
        adaptive = 0.01 * min(recent_err, 5.0)  # cap to avoid instability
        q_scalar = base + adaptive
        
        return q_scalar * np.eye(self.d)

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.time_limit:
            # Return the arm with highest current estimate
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        for a in range(self.n):
            # Predict step: inflate covariance with process noise
            Q_a = self._adaptive_Q(a)
            P_pred = self.P[a] + Q_a
            
            # Estimate
            theta = self.theta[a]
            pred_reward = theta @ context
            
            # UCB score: mean + beta * uncertainty
            uncertainty = np.sqrt(context @ P_pred @ context)
            score = pred_reward + self.beta * uncertainty
            
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() - self.start_time > self.time_limit:
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        # Get current state for this arm
        theta = self.theta[arm]
        P = self.P[arm]

        # Predict step
        Q_a = self._adaptive_Q(arm)
        P_pred = P + Q_a

        # Prediction
        pred_reward = theta @ context

        # Innovation (prediction error)
        innovation = reward - pred_reward

        # Track prediction error for adaptive Q
        self.pred_errors[arm] = 0.9 * self.pred_errors[arm] + 0.1 * abs(innovation)
        self.error_count[arm] += 1

        # Kalman gain
        # S = innovation covariance
        S = context @ P_pred @ context + 1.0  # +1 for observation noise
        K = P_pred @ context / S  # gain vector

        # Update
        self.theta[arm] = theta + K * innovation
        # Joseph form for numerical stability
        I_K_ctx = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_K_ctx @ P_pred @ I_K_ctx.T + K * K.T * 1.0  # + K K^T * R, R=1

        # Clamp covariance to prevent numerical drift
        self.P[arm] = np.maximum(self.P[arm], 1e-8 * np.eye(self.d))

        self.count[arm] += 1
# EVOLVE-BLOCK-END
