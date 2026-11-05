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
        self.deadline = 135.0  # tighter safety margin before 150s kill

        # Inverse covariance matrices V = A^{-1} for each arm
        # Start with ridge for numerical stability
        ridge = 0.15
        self.V = [np.eye(dim) * (1.0 + ridge) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.95
        self.gamma_min = 0.75
        self.gamma_max = 0.995
        self.C = 1.0  # exploration coefficient
        self.C_min = 0.2
        self.C_max = 4.0

        # For tracking prediction errors to adapt forgetting
        self.pred_errors = [0.0] * n_arms
        self.error_counts = [0] * n_arms

        # Time tracking for schedule
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on recent prediction error rate."""
        c = self.error_counts[arm]
        if c < 5:
            return self.base_gamma

        # Use recent prediction error to adjust gamma
        # Normalize by count to get mean squared error
        mse = self.pred_errors[arm] / c
        # Use sqrt to get RMS error for more stable adaptation
        rms_err = np.sqrt(mse)
        # Scale error to adjust gamma: higher error -> lower gamma (faster forgetting)
        # Use a smoother scaling that responds to moderate errors
        err_scale = min(rms_err, 3.0)
        # More aggressive adaptation: up to 0.15 reduction in gamma
        gamma = self.base_gamma - 0.15 * err_scale
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm):
        """Adapt exploration based on count with faster early decay."""
        c = self.counts[arm]
        if c < 2:
            return self.C_max
        if c < 5:
            return self.C * 2.0
        if c < 15:
            # Faster decay in mid-range
            return self.C * (1.5 - 0.1 * (c - 5))
        # Long-term: gentle time-based decay
        time_decay = max(0.3, 1.0 - 0.05 * (self.t / 300.0))
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
            # UCB bonus with count-aware scaling
            c_val = self._adaptive_C(a)
            # Add small floor to avoid zero bonus for well-explored arms
            variance = context @ self.V[a] @ context
            bonus = c_val * np.sqrt(variance + 1e-8)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error for adaptive gamma
        # Use current estimate before update
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = reward - pred
        # Use exponential moving average for more responsive error tracking
        # Weight recent errors more heavily
        self.pred_errors[arm] = 0.9 * self.pred_errors[arm] + 0.1 * err * err
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

        # Periodic numerical stability check: if V becomes ill-conditioned, renormalize
        if self.t % 100 == 0:
            # Check condition number
            try:
                eigvals = np.linalg.eigvalsh(self.V[arm])
                if eigvals.min() < 1e-10 or eigvals.max() > 1e10:
                    # Reset to a well-conditioned state
                    self.V[arm] = np.eye(self.d) * (1.0 + 0.15)
            except:
                pass
# EVOLVE-BLOCK-END
