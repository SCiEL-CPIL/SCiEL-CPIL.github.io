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
        self.deadline = 115.0  # safety margin before 120s evaluator limit

        # Inverse covariance matrices V = A^{-1} for each arm
        # Start with slightly larger ridge for better numerical stability
        ridge = 0.15
        self.V = [np.eye(dim) * (1.0 + ridge) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters — tuned forgetting for drift tracking
        self.base_gamma = 0.94
        self.gamma_min = 0.75
        self.gamma_max = 0.985
        self.C = 1.3  # exploration coefficient
        self.C_min = 0.15
        self.C_max = 4.5

        # For tracking prediction errors to adapt forgetting
        # Use exponential moving average for faster reaction
        self.pred_err_ema = [0.0] * n_arms
        self.err_ema_alpha = 0.30  # balanced EMA: fast enough to react, smooth enough to avoid noise

        # Time tracking for schedule
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on recent prediction error (EMA)."""
        c = self.counts[arm]
        if c < 3:
            return self.base_gamma

        # Use EMA of squared prediction error to adjust gamma
        # Higher recent error -> faster forgetting (lower gamma)
        err_ema = self.pred_err_ema[arm]
        # Smooth logistic-like mapping: err_ema=0 -> base_gamma, err_ema=2+ -> near gamma_min
        # Using a smooth sigmoid for differentiability and monotonicity
        err_scale = min(err_ema, 6.0)
        # Logistic mapping: 1/(1+exp(-k*(x-x0))) with x0=0.5, k=1.2
        sig = 1.0 / (1.0 + np.exp(-1.2 * (err_scale - 0.5)))
        gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * sig
        # Blend with base_gamma for stability when error is low
        blend = 0.3 * (1.0 - sig)
        gamma = blend * self.base_gamma + (1.0 - blend) * gamma
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm):
        """Adapt exploration based on count and time with smooth two-phase decay."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 6:
            return self.C * 1.4
        # Two-phase: initial fast decay, then slow floor to maintain exploration
        # Phase 1: exp decay over first 500 steps
        # Phase 2: linear floor to C_min
        t = float(self.t)
        if t < 500.0:
            time_decay = np.exp(-t / 300.0)
            time_decay = max(0.4, time_decay)
        else:
            # Slow linear decay from 0.4 to C_min/C over remaining time
            frac = min((t - 500.0) / 2000.0, 1.0)
            time_decay = 0.4 - frac * (0.4 - self.C_min / self.C)
            time_decay = max(self.C_min / self.C, time_decay)
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

        # Compute prediction error for adaptive gamma
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = reward - pred
        # Update EMA of squared error
        self.pred_err_ema[arm] = (1.0 - self.err_ema_alpha) * self.pred_err_ema[arm] + self.err_ema_alpha * (err * err)

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

        # Smooth numerical stabilization: continuous small ridge to prevent drift
        # Add a tiny ridge every step proportional to 1/t for diminishing effect
        ridge_step = 0.0008 / (1.0 + self.t * 0.01)
        if self.t % 10 == 0:
            for a in range(self.n):
                self.V[a] += ridge_step * np.eye(self.d)
# EVOLVE-BLOCK-END
