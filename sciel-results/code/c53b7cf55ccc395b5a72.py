# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm Kalman filter with adaptive process noise
    - UCB exploration with adaptive coefficient
    - Prediction-residual-driven process noise adaptation
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before 120s kill

        # Kalman filter state per arm
        # P: covariance of theta estimate
        # theta: mean of theta estimate
        # Q: process noise covariance (scalar * I for simplicity)
        ridge = 1.0
        self.P = [np.eye(dim) * (10.0 + ridge) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Base process noise level
        self.base_Q = 0.01
        # Exploration coefficient
        self.base_alpha = 1.2
        # Track prediction errors for adaptive process noise
        self.residuals = [0.0] * n_arms
        self.residual_decay = 0.9
        # Measurement noise
        self.R = 1.0
        
        # Count steps to adapt alpha
        self.step_count = 0
        self.warmup_steps = 40

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_Q(self, arm, residual):
        """Adapt process noise based on prediction residual."""
        # Update residual tracker (EMA of squared residuals)
        self.residuals[arm] = self.residual_decay * self.residuals[arm] + (1 - self.residual_decay) * residual ** 2
        
        # Normalize residual to drive Q
        # Larger residuals → larger Q (more process noise, faster adaptation)
        # Scale: typical residual^2 might be 1-10, so normalize
        res_norm = min(self.residuals[arm] / 5.0, 2.0)
        # Q between base_Q and 10*base_Q
        Q = self.base_Q * (1.0 + 9.0 * res_norm)
        return Q

    def _adaptive_alpha(self):
        """Adapt exploration coefficient."""
        if self.step_count < self.warmup_steps:
            return self.base_alpha * 1.5
        else:
            factor = 1.0 / (1.0 + self.step_count / 600.0)
            return self.base_alpha * max(factor, 0.5)

    def select(self, context):
        if self._check_deadline():
            return int(np.random.randint(0, self.n))
        
        self.step_count += 1
        alpha = self._adaptive_alpha()
        
        scores = []
        for a in range(self.n):
            try:
                pred = self.theta[a] @ context
                # UCB bonus using Kalman covariance
                var = context @ self.P[a] @ context
                bonus = alpha * np.sqrt(var)
                scores.append(pred + bonus)
            except np.linalg.LinAlgError:
                scores.append(np.random.randn())
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return
        
        x = context
        r = reward
        
        # Compute prediction and residual
        pred = self.theta[arm] @ x
        residual = r - pred
        
        # Adaptive process noise
        Q = self._adaptive_Q(arm, residual)
        
        # Kalman filter update
        # Predict step (process noise)
        P_pred = self.P[arm] + Q * np.eye(self.d)
        
        # Measurement update
        # S = x^T P_pred x + R
        S = x @ P_pred @ x + self.R
        # K = P_pred x / S
        K = P_pred @ x / S
        # Update theta
        self.theta[arm] = self.theta[arm] + K * residual
        # Update P (Joseph form for numerical stability)
        I_Kx = np.eye(self.d) - np.outer(K, x)
        self.P[arm] = I_Kx @ P_pred @ I_Kx.T + self.R * np.outer(K, K)
# EVOLVE-BLOCK-END
