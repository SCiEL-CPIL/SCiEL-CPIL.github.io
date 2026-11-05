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
        # Start with slight ridge for numerical stability
        ridge = 0.1
        self.V = [np.eye(dim) * (1.0 + ridge) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.965
        self.gamma_min = 0.85
        self.gamma_max = 0.997
        self.C = 1.2  # exploration coefficient
        self.C_min = 0.3
        self.C_max = 5.0

        # For tracking prediction errors to adapt forgetting
        # Use exponential moving average for faster reaction
        self.pred_err_ema = [0.0] * n_arms
        self.err_ema_alpha = 0.25  # EMA decay for error tracking
        # Track relative error (normalized by reward scale)
        self.reward_scale_ema = [1.0] * n_arms
        self.reward_scale_alpha = 0.1

        # Time tracking for schedule
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on relative prediction error (EMA)."""
        c = self.counts[arm]
        if c < 5:
            return self.base_gamma

        # Use normalized error for more stable adaptation
        err_ema = self.pred_err_ema[arm]
        reward_scale = max(self.reward_scale_ema[arm], 0.1)
        # Relative error: how large is prediction error compared to reward scale
        rel_err = err_ema / reward_scale
        # Cap to avoid extreme values
        rel_err = min(rel_err, 5.0)
        
        # Smooth, monotonic mapping using sigmoid-like function
        # rel_err ~ 0 -> gamma close to base_gamma
        # rel_err ~ 1+ -> gamma drops toward gamma_min
        # Use tanh for smooth saturation
        gamma_shift = 0.12 * np.tanh(0.8 * rel_err)
        gamma = self.base_gamma - gamma_shift
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm):
        """Adapt exploration based on count, uncertainty, and time with smooth decay."""
        c = self.counts[arm]
        if c < 5:
            return self.C_max
        if c < 10:
            return self.C * 1.3
        
        # Combine time-based decay with uncertainty-based scaling
        time_decay = np.exp(-self.t / 1200.0)
        time_decay = max(0.2, time_decay)
        
        # Scale exploration by how uncertain we are about this arm
        # Use trace of V as uncertainty proxy (normalized)
        V_trace = np.trace(self.V[arm])
        base_trace = self.d * (1.0 + 0.1)  # initial trace value
        uncertainty_ratio = min(V_trace / base_trace, 3.0)
        
        # Final C: base * time_decay * uncertainty_factor
        uncertainty_factor = 0.5 + 0.5 * uncertainty_ratio
        return self.C * time_decay * uncertainty_factor

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

        # Compute prediction error for adaptive gamma
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = reward - pred
        # Update EMA of squared error
        self.pred_err_ema[arm] = (1.0 - self.err_ema_alpha) * self.pred_err_ema[arm] + self.err_ema_alpha * (err * err)
        
        # Track reward scale for normalization
        abs_reward = abs(reward)
        self.reward_scale_ema[arm] = (1.0 - self.reward_scale_alpha) * self.reward_scale_ema[arm] + self.reward_scale_alpha * abs_reward

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
            self.V[arm] = np.eye(self.d) * (1.0 + 0.1)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Periodic numerical stabilization: every 150 steps, add adaptive ridge
        if self.t % 150 == 0:
            for a in range(self.n):
                # Add ridge proportional to current matrix scale
                V_scale = np.trace(self.V[a]) / self.d
                ridge_add = max(0.0005, 0.001 * V_scale)
                self.V[a] += ridge_add * np.eye(self.d)
# EVOLVE-BLOCK-END
