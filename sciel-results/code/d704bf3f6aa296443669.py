"""Non-stationary linear contextual bandit with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least-squares with adaptive forgetting
    - UCB exploration with time-decaying coefficient
    - Intercept augmentation for mean reward capture
    - Prediction-error-driven adaptive forgetting
    - Ridge regularization for numerical stability
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d_aug = dim + 1  # augmented dimension (with intercept)
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0

        # Per-arm state (augmented)
        self.V = [np.eye(self.d_aug) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(self.d_aug) for _ in range(n_arms)]
        self.counts = [0] * n_arms
        self.theta = [np.zeros(self.d_aug) for _ in range(n_arms)]

        # Global adaptive parameters
        self.gamma = 0.97
        self.gamma_min = 0.88
        self.gamma_max = 0.999
        self.lambda_ridge = 0.1

        # Exploration
        self.C_base = 2.5
        self.C_min = 0.8
        self.C_max = 5.0
        self.round_num = 0

        # Prediction error tracking
        self.pred_errors = []
        self.pred_error_window_max = 50
        self.base_pred_error = None

        # Per-arm reward stats
        self.arm_reward_sums = [0.0] * n_arms
        self.arm_reward_sq_sums = [0.0] * n_arms
        self.arm_pred_err_sums = [0.0] * n_arms
        self.arm_pred_err_sq_sums = [0.0] * n_arms

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _augment(self, context):
        """Augment context with intercept term (1.0)."""
        x = np.empty(self.d_aug)
        x[:-1] = context
        x[-1] = 1.0
        return x

    def _get_C(self, arm):
        """Time-decaying exploration coefficient with per-arm boost."""
        c = self.counts[arm]
        t = self.round_num
        # Smooth exponential decay
        decay = np.exp(-t / 80.0)
        C_global = self.C_min + (self.C_max - self.C_min) * decay

        # Per-arm boost for under-explored arms
        if c == 0:
            C_arm = self.C_max
        elif c < 5:
            C_arm = C_global * 2.0
        elif c < 15:
            C_arm = C_global * 1.3
        else:
            C_arm = C_global

        return min(C_arm, self.C_max)

    def _adaptive_gamma(self, arm):
        """
        Adapt forgetting factor based on:
        1. Global prediction error trend
        2. Per-arm recent prediction error
        """
        gamma = self.gamma
        c = self.counts[arm]

        # Global prediction error signal
        if len(self.pred_errors) >= 10:
            recent = self.pred_errors[-15:]
            mean_recent = np.mean(recent)
            if self.base_pred_error is not None and self.base_pred_error > 1e-8:
                ratio = mean_recent / self.base_pred_error
                if ratio > 1.3:
                    gamma -= 0.04 * min(ratio - 1.0, 2.5)
                elif ratio < 0.6:
                    gamma += 0.02 * (1.0 - ratio)

        # Per-arm prediction error signal
        if c >= 8:
            # Use recent window of per-arm errors
            arm_mean_err = self.arm_pred_err_sums[arm] / c
            arm_var_err = max(0.0, self.arm_pred_err_sq_sums[arm] / c - arm_mean_err ** 2)
            # High variance in prediction errors suggests non-stationarity
            err_scale = min(arm_mean_err, 3.0)
            gamma -= 0.025 * err_scale

        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _update_theta(self, arm):
        self.theta[arm] = self.V[arm] @ self.b[arm]

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        x = self._augment(context)

        # Deadline guard: greedy fallback
        if not self._check_deadline():
            scores = [self.theta[a] @ x for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ x
            c_val = self._get_C(a)
            Vx = self.V[a] @ x
            var_term = x @ Vx
            if var_term < 0:
                var_term = 0.0
            bonus = c_val * np.sqrt(var_term)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        x = self._augment(context)
        self.round_num += 1

        # Track prediction error
        pred = self.theta[arm] @ x
        pred_err = abs(reward - pred)
        self.pred_errors.append(pred_err)
        if len(self.pred_errors) > self.pred_error_window_max:
            self.pred_errors.pop(0)
        if len(self.pred_errors) == 15 and self.base_pred_error is None:
            self.base_pred_error = max(np.mean(self.pred_errors), 1e-8)

        # Track per-arm stats
        self.counts[arm] += 1
        self.arm_reward_sums[arm] += reward
        self.arm_reward_sq_sums[arm] += reward * reward
        self.arm_pred_err_sums[arm] += pred_err
        self.arm_pred_err_sq_sums[arm] += pred_err * pred_err

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Sherman-Morrison update for V = A^{-1}
        # V_new = (1/gamma) * V_old - (1/gamma) * (V_old x)(V_old x)^T / (gamma + x^T V_old x)
        V_old = self.V[arm]
        Vx = V_old @ x
        xVx = x @ Vx
        denom = gamma + xVx

        if denom > 1e-10:
            V_new = (1.0 / gamma) * V_old - (1.0 / (gamma * denom)) * np.outer(Vx, Vx)
        else:
            V_new = np.eye(self.d_aug) * (1.0 / self.lambda_ridge)

        # Symmetrize
        V_new = 0.5 * (V_new + V_new.T)

        # Numerical stability: clip diagonal
        diag = np.diag(V_new)
        if np.any(diag < 1e-10):
            V_new = np.eye(self.d_aug) * (1.0 / self.lambda_ridge)

        # Also check for large values (instability)
        if np.max(np.abs(V_new)) > 1e6:
            V_new = np.eye(self.d_aug) * (1.0 / self.lambda_ridge)

        self.V[arm] = V_new

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * x

        # Update theta
        self._update_theta(arm)
# EVOLVE-BLOCK-END
