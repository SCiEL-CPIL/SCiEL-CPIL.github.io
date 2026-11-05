"""Non-stationary linear contextual bandit with shared model and arm-specific basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Shared linear model with arm-specific one-hot basis functions
    - Discounted least-squares with adaptive forgetting on the unified model
    - UCB exploration with time-decaying coefficient
    - Prediction-error-driven acceleration of forgetting
    - Adaptive ridge regularization
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Extended dimension: context + one-hot arm indicators
        self.d_ext = dim + n_arms
        self.start_time = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

        # Unified model state
        self.V = np.eye(self.d_ext) * 1.0  # inverse covariance for extended feature
        self.b = np.zeros(self.d_ext)       # weighted sum of rewards * extended_feature
        self.theta = np.zeros(self.d_ext)   # cached theta = V @ b
        self.counts = [0] * n_arms

        # Global adaptive parameters
        self.gamma = 0.97          # base forgetting factor
        self.gamma_min = 0.85
        self.gamma_max = 0.999
        self.lambda_ridge = 0.1    # ridge regularization

        # Exploration: starts aggressive, decays
        self.C_base = 2.0
        self.C_min = 0.5
        self.C_max = 4.0
        self.round_num = 0

        # Per-arm prediction error tracking for adaptive gamma
        self.arm_pred_errors = [[] for _ in range(n_arms)]
        self.pred_error_window_max = 30
        self.base_pred_errors = [None] * n_arms

        # Per-arm recent reward stats for variance-based gamma
        self.arm_reward_sums = [0.0] * n_arms
        self.arm_reward_sq_sums = [0.0] * n_arms

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _get_extended_feature(self, context, arm):
        """Build extended feature vector: [context; one-hot arm indicator]."""
        ext = np.zeros(self.d_ext, dtype=np.float64)
        ext[:self.d] = context
        ext[self.d + arm] = 1.0
        return ext

    def _get_C(self, arm):
        """
        Exploration coefficient with visit scaling.
        Scales down as count increases to reduce over-exploration of well-known arms.
        """
        c = self.counts[arm]
        t = self.round_num
        
        # Base decay
        decay = 1.0 / (1.0 + t / 100.0)
        C_global = self.C_min + (self.C_max - self.C_min) * decay
        
        # Visit scaling: reduce bonus for highly sampled arms
        if c == 0:
            scale = 1.0
        else:
            scale = 1.0 / np.sqrt(1.0 + c)
            
        C_arm = C_global * scale
        
        # Ensure at least some exploration for new arms
        if c < 2:
            C_arm = max(C_arm, self.C_max)
            
        return min(C_arm, self.C_max)

    def _adaptive_gamma(self, arm):
        """
        Adapt forgetting factor based on:
        1. Per-arm prediction error trend (spike -> faster forgetting)
        2. Per-arm reward variance (high variance -> faster forgetting)
        """
        c = self.counts[arm]

        # Start with base gamma
        gamma = self.gamma

        # Per-arm prediction error signal
        if len(self.arm_pred_errors[arm]) >= 5:
            recent_errors = np.array(self.arm_pred_errors[arm][-10:])
            mean_err = np.mean(recent_errors)
            if self.base_pred_errors[arm] is not None and self.base_pred_errors[arm] > 1e-8:
                ratio = mean_err / self.base_pred_errors[arm]
                # If errors are growing, accelerate forgetting more aggressively
                if ratio > 1.5:
                    gamma -= 0.15 * min(ratio - 1.0, 2.0)
                elif ratio < 0.7:
                    # Errors shrinking, slow down forgetting
                    gamma += 0.05 * (1.0 - ratio)

        # Per-arm variance signal
        if c >= 5:
            mean_r = self.arm_reward_sums[arm] / c
            var_r = max(0.0, self.arm_reward_sq_sums[arm] / c - mean_r ** 2)
            var_scale = min(var_r, 5.0)
            gamma -= 0.05 * var_scale

        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _update_theta(self):
        """Update cached theta for the unified model."""
        self.theta = self.V @ self.b

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard
        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                ext = self._get_extended_feature(context, a)
                scores.append(self.theta @ ext)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            ext = self._get_extended_feature(context, a)
            mu = self.theta @ ext
            c_val = self._get_C(a)
            # UCB bonus: c * sqrt(z^T V z) where z is extended feature
            Vz = self.V @ ext
            var_term = ext @ Vz
            if var_term < 0:
                var_term = 0.0
            bonus = c_val * np.sqrt(var_term)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1

        # Build extended feature
        ext = self._get_extended_feature(context, arm)

        # Track prediction error for adaptive gamma
        pred = self.theta @ ext
        pred_err = abs(reward - pred)
        self.arm_pred_errors[arm].append(pred_err)
        if len(self.arm_pred_errors[arm]) > self.pred_error_window_max:
            self.arm_pred_errors[arm].pop(0)
        if len(self.arm_pred_errors[arm]) == 10 and self.base_pred_errors[arm] is None:
            self.base_pred_errors[arm] = max(np.mean(self.arm_pred_errors[arm]), 1e-8)

        # Track per-arm reward stats
        self.counts[arm] += 1
        self.arm_reward_sums[arm] += reward
        self.arm_reward_sq_sums[arm] += reward * reward

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance with forgetting using Sherman-Morrison
        # V_new = (gamma * V_old^{-1} + z z^T)^{-1}
        # = (1/gamma) * V_old - (1/gamma) * V_old z z^T V_old / (gamma + z^T V_old z)

        V_old = self.V
        Vz = V_old @ ext
        zVz = ext @ Vz  # scalar
        denom = gamma + zVz

        if denom > 1e-10:
            V_new = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vz, Vz) / denom
        else:
            V_new = np.eye(self.d_ext) * (1.0 / self.lambda_ridge)

        # Symmetrize for numerical stability
        V_new = 0.5 * (V_new + V_new.T)

        # Clip: ensure positive semi-definite by adding small ridge if needed
        diag = np.diag(V_new)
        if np.any(diag < 1e-8):
            V_new = np.eye(self.d_ext) * (1.0 / self.lambda_ridge)

        self.V = V_new

        # Update b with forgetting
        self.b = gamma * self.b + reward * ext

        # Update theta
        self._update_theta()
# EVOLVE-BLOCK-END
