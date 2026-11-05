"""Bootstrap spawn: direct approach for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm ridge regression with exponential forgetting (discounted least squares)
    - Smooth adaptive forgetting rate based on log-ratio of prediction residuals
    - UCB-style exploration with arm-specific uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s kill

        # Per-arm parameters: V = (sum gamma^t x x^T + lambda I)^{-1}, b = sum gamma^t r x
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # For adaptive forgetting
        self.prediction_errors = [[] for _ in range(n_arms)]
        self.max_err_window = 50

        # Exploration
        self.C = 2.0
        self.base_gamma = 0.93  # Slightly faster base forgetting for non-stationarity
        self.gamma_min = 0.84   # Allow deeper forgetting on regime change
        self.gamma_max = 0.99

        # Track last few contexts for possible restarts
        self.recent_contexts = []
        self.max_recent = 50

        # Global model for cross-arm sharing
        self.d_global = dim + n_arms
        self.V_global = np.eye(self.d_global)
        self.b_global = np.zeros(self.d_global)
        self.counts_global = 0
        self.prediction_errors_global = []
        self.max_err_window_global = 50
        self.base_gamma_global = 0.93
        self.gamma_min_global = 0.84
        self.gamma_max_global = 0.99

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor using smooth log-ratio of recent vs older errors."""
        c = self.counts[arm]
        if c < 5:
            return self.base_gamma

        recent_errs = self.prediction_errors[arm]
        if len(recent_errs) < 10:
            return self.base_gamma

        n = len(recent_errs)
        # Use last 10 for recent, all before that for older
        recent_mean = np.mean(recent_errs[-10:])
        older_mean = np.mean(recent_errs[:max(1, n - 10)])

        if older_mean < 1e-12:
            log_ratio = 0.0
        else:
            ratio = recent_mean / older_mean
            log_ratio = np.log(max(ratio, 1e-6))

        # Smooth mapping: negative adjustment when errors increase (log_ratio > 0)
        # Scale factor tuned for responsiveness without instability
        adjustment = -1.8 * log_ratio
        gamma = self.base_gamma + adjustment

        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_gamma_global(self):
        """Adapt forgetting factor for global model."""
        c = self.counts_global
        if c < 5:
            return self.base_gamma_global

        recent_errs = self.prediction_errors_global
        if len(recent_errs) < 10:
            return self.base_gamma_global

        n = len(recent_errs)
        recent_mean = np.mean(recent_errs[-10:])
        older_mean = np.mean(recent_errs[:max(1, n - 10)])

        if older_mean < 1e-12:
            log_ratio = 0.0
        else:
            ratio = recent_mean / older_mean
            log_ratio = np.log(max(ratio, 1e-6))

        adjustment = -1.8 * log_ratio
        gamma = self.base_gamma_global + adjustment

        return max(self.gamma_min_global, min(self.gamma_max_global, gamma))

    def _adaptive_C(self, arm):
        """Adapt exploration coefficient based on count with smooth transitions."""
        c = self.counts[arm]
        if c < 3:
            return self.C * 2.5
        if c < 8:
            return self.C * 1.8
        if c < 20:
            return self.C * 1.3
        if c < 50:
            return self.C * 1.0
        return self.C

    def _augment_context(self, context, arm):
        """Create augmented context with one-hot arm indicator."""
        aug = np.zeros(self.d_global)
        aug[:self.d] = context
        aug[self.d + arm] = 1.0
        return aug

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        # UCB with adaptive exploration, combining per-arm and global model
        scores = []
        for a in range(self.n):
            # Per-arm model
            theta_arm = self.V[a] @ self.b[a]
            mu_arm = theta_arm @ context
            c_val_arm = self._adaptive_C(a)
            bonus_arm = c_val_arm * np.sqrt(context @ self.V[a] @ context)

            # Global model
            aug_ctx = self._augment_context(context, a)
            theta_global = self.V_global @ self.b_global
            mu_global = theta_global @ aug_ctx
            c_val_global = self.C
            bonus_global = c_val_global * np.sqrt(aug_ctx @ self.V_global @ aug_ctx)

            # Blend: weight by count
            total_count = self.counts[a] + self.counts_global
            if total_count == 0:
                mu = mu_arm
                bonus = bonus_arm
            else:
                w_arm = self.counts[a] / total_count
                w_global = self.counts_global / total_count
                mu = w_arm * mu_arm + w_global * mu_global
                bonus = w_arm * bonus_arm + w_global * bonus_global

            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1
        self.counts_global += 1

        # Compute prediction error before updating (per-arm)
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = abs(reward - pred)
        self.prediction_errors[arm].append(err)
        if len(self.prediction_errors[arm]) > self.max_err_window:
            self.prediction_errors[arm].pop(0)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V[arm] = np.eye(self.d)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Periodic regularization to prevent numerical drift
        if self.counts[arm] % 100 == 0:
            self.V[arm] += 1e-8 * np.eye(self.d)

        # Update global model
        aug_ctx = self._augment_context(context, arm)
        
        # Compute prediction error for global model
        theta_global = self.V_global @ self.b_global
        pred_global = theta_global @ aug_ctx
        err_global = abs(reward - pred_global)
        self.prediction_errors_global.append(err_global)
        if len(self.prediction_errors_global) > self.max_err_window_global:
            self.prediction_errors_global.pop(0)

        # Adaptive forgetting for global
        gamma_global = self._adaptive_gamma_global()

        # Update global inverse covariance
        Vx_global = self.V_global @ aug_ctx
        xVx_global = aug_ctx @ Vx_global
        denom_global = gamma_global + xVx_global

        if denom_global > 1e-12:
            self.V_global = (1.0 / gamma_global) * self.V_global - (1.0 / gamma_global) * np.outer(Vx_global, Vx_global) / denom_global
        else:
            self.V_global = np.eye(self.d_global)

        # Update global b
        self.b_global = gamma_global * self.b_global + reward * aug_ctx

        # Periodic regularization for global
        if self.counts_global % 100 == 0:
            self.V_global += 1e-8 * np.eye(self.d_global)
# EVOLVE-BLOCK-END
