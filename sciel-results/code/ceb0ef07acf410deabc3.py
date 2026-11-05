"""Non-stationary linear contextual bandit with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least-squares with adaptive forgetting
    - UCB exploration with adaptive coefficient
    - Prediction-error-driven acceleration of forgetting
    - Ridge regularization for numerical stability
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

        # Inverse covariance matrices V = A^{-1} for each arm
        # Slightly larger initial inverse for more exploration early on
        self.V = [np.eye(dim) * 1.5 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.96
        self.gamma_min = 0.82
        self.gamma_max = 0.995
        self.C = 1.1  # exploration coefficient
        self.C_min = 0.2
        self.C_max = 4.0

        # For tracking prediction errors to adapt forgetting (EMA)
        self.ema_sq_err = [0.0] * n_arms
        self.ema_alpha = 0.1  # EMA decay for error tracking
        self.error_counts = [0] * n_arms

        # Time tracking for schedule
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on EMA of prediction error."""
        c = self.error_counts[arm]
        if c < 5:
            return self.base_gamma

        # Use EMA of squared prediction error to adjust gamma
        # Higher prediction error -> faster forgetting (lower gamma)
        # Use sqrt of EMA to get RMSE-like scale
        rmse = np.sqrt(self.ema_sq_err[arm])
        # Smoothly map RMSE to gamma adjustment
        err_scale = min(rmse, 5.0)
        gamma = self.base_gamma - 0.05 * err_scale
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm):
        """Adapt exploration based on count and time with smoother decay."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 10:
            return self.C * 1.5
        # Smooth exponential decay of exploration over time
        time_decay = np.exp(-0.05 * (self.t / 500.0))
        time_decay = max(0.4, time_decay)
        return self.C * time_decay

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            # UCB bonus
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(context @ self.V[a] @ context)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error for adaptive gamma (EMA update)
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = reward - pred
        sq_err = err * err
        self.ema_sq_err[arm] = (1.0 - self.ema_alpha) * self.ema_sq_err[arm] + self.ema_alpha * sq_err
        self.error_counts[arm] += 1

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        # V_new = (gamma * V_old^{-1} + x x^T)^{-1}
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            # Fallback: reset to identity scaled
            self.V[arm] = np.eye(self.d)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
