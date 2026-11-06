import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive LinUCB with global linear model for non-stationary linear contextual bandits.
    Uses a single global model with arm identity features to share context representations.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.ext_dim = dim + n_arms
        
        # Global model state
        self.A = np.eye(self.ext_dim)
        self.b = np.zeros(self.ext_dim)
        self.inv_A = np.eye(self.ext_dim)
        
        # Per-arm counters for exploration decay
        self.arm_counts = np.zeros(n_arms)
        
        # Adaptive Forgetting Parameters
        self.lambda_min = 0.85
        self.lambda_max = 0.995
        self.error_window = deque(maxlen=30)
        
        # Exploration
        self.alpha_base = 1.5
        self.alpha_min = 0.5
        self.alpha_max = 3.0
        self.alpha = self.alpha_base
        
        # Timing
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 120s limit
        
        # Global step
        self.t = 0
        
        # Global error tracking for alpha adaptation
        self.global_errors = deque(maxlen=50)

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _get_context(self, context, arm):
        """Create extended context with arm identity one-hot features."""
        ext = np.zeros(self.ext_dim)
        ext[:self.d] = context
        ext[self.d + arm] = 1.0
        return ext

    def _update_inv_woodbury(self, ext_context, lam):
        """
        Update inverse using Woodbury identity for A_new = lam * A_old + x x^T.
        """
        inv_old = self.inv_A
        # M^{-1} = (1/lam) * inv_old
        inv_M = inv_old / lam
        
        # Compute v = M^{-1} x
        v = inv_M @ ext_context
        
        # Compute denominator: 1 + x^T v
        denom = 1.0 + ext_context @ v
        
        # Update inverse
        self.inv_A = inv_M - np.outer(v, v) / denom

    def _get_lambda(self):
        """Determine forgetting factor based on recent errors."""
        if len(self.error_window) == 0:
            return self.lambda_max
        
        errors = np.array(self.error_window)
        mse = np.mean(errors**2)
        
        # Use a sharper transition
        scale = 0.02
        norm_mse = mse / (mse + scale)
        lam = self.lambda_max - norm_mse * (self.lambda_max - self.lambda_min)
        
        return max(self.lambda_min, min(self.lambda_max, lam))

    def _update_alpha(self):
        """Dynamically adjust alpha based on global performance."""
        if len(self.global_errors) < 5:
            return
        
        errors = np.array(self.global_errors)
        mse = np.mean(errors**2)
        
        # Consider trend
        if len(errors) >= 10:
            half = len(errors) // 2
            recent_half_mse = np.mean(errors[half:]**2)
            older_half_mse = np.mean(errors[:half]**2)
            trend = recent_half_mse - older_half_mse
        else:
            trend = 0.0
        
        # Base target from MSE
        target_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.1))
        
        # Adjust for trend
        if trend > 0.01:
            target_alpha *= 1.15
        elif trend < -0.01:
            target_alpha *= 0.95
        
        # Smoothly move towards target
        self.alpha = 0.9 * self.alpha + 0.1 * target_alpha
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def select(self, context):
        if not self._check_deadline():
            # If we are near the deadline, just pick the arm with highest mean prediction
            scores = []
            for a in range(self.n):
                try:
                    ext_ctx = self._get_context(context, a)
                    theta = self.inv_A @ self.b
                    mu = theta @ ext_ctx
                except:
                    mu = 0.0
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            ext_ctx = self._get_context(context, a)
            theta = self.inv_A @ self.b
            mu = theta @ ext_ctx
            var = ext_ctx @ self.inv_A @ ext_ctx
            # Count-based exploration decay: reduce bonus for well-sampled arms
            count_factor = 1.0 / (1.0 + 0.1 * self.arm_counts[a])
            bonus = self.alpha * count_factor * np.sqrt(var)
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        self.t += 1
        self.arm_counts[arm] += 1
        
        # Create extended context
        ext_ctx = self._get_context(context, arm)
        
        # 1. Calculate residual using the model before this update
        theta_old = self.inv_A @ self.b
        pred = theta_old @ ext_ctx
        residual = reward - pred
        
        # Store residual
        self.error_window.append(residual)
        self.global_errors.append(residual)
        
        # 2. Determine current forgetting factor
        lam = self._get_lambda()
        
        # 3. Update matrices
        self.A = lam * self.A + np.outer(ext_ctx, ext_ctx)
        self.b = lam * self.b + reward * ext_ctx
        
        # Update inverse using Woodbury
        self._update_inv_woodbury(ext_ctx, lam)
        
        # 4. Update alpha
        self._update_alpha()


