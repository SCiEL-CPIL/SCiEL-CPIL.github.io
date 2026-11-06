import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive LinUCB for non-stationary linear contextual bandits.
    Combines:
    - Structural idea: Per-arm adaptive forgetting factor (lambda) based on local MSE.
    - Refinement machinery: Global adaptive alpha (exploration) based on global MSE,
      and a strict deadline guard for robustness.
    - Cached inverses for speed.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]  # Cached inverses
        
        # Adaptive Forgetting Parameters
        self.lambda_min = 0.88
        self.lambda_max = 0.998
        self.error_windows = [deque(maxlen=20) for _ in range(n_arms)]
        
        # Exploration
        self.alpha_base = 1.0
        self.alpha_min = 0.3
        self.alpha_max = 2.5
        self.alpha = self.alpha_base
        
        # Timing
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 120s limit
        
        # Global step
        self.t = 0
        
        # Global error tracking for alpha adaptation
        self.global_errors = deque(maxlen=50)
        
        # Short-term reward tracking for regime change detection
        self.recent_rewards = deque(maxlen=15)
        self.reward_mean = 0.0
        self.reward_std = 1.0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _recompute_inv(self, arm):
        """Recompute inverse for arm's A matrix."""
        try:
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            self.inv_A[arm] = np.eye(self.d)

    def _get_lambda(self, arm):
        """Determine forgetting factor based on recent errors for this arm."""
        if len(self.error_windows[arm]) == 0:
            return self.lambda_max
        
        errors = np.array(self.error_windows[arm])
        mse = np.mean(errors**2)
        
        # Use a sharper transition: smaller scale means faster response to error
        scale = 0.05
        norm_mse = mse / (mse + scale)
        lam = self.lambda_max - norm_mse * (self.lambda_max - self.lambda_min)
        
        return max(self.lambda_min, min(self.lambda_max, lam))

    def _update_alpha(self):
        """Dynamically adjust alpha based on global performance."""
        if len(self.global_errors) < 5:
            return
        
        errors = np.array(self.global_errors)
        mse = np.mean(errors**2)
        
        # Also consider trend: are errors increasing or decreasing?
        if len(errors) >= 10:
            half = len(errors) // 2
            recent_half_mse = np.mean(errors[half:]**2)
            older_half_mse = np.mean(errors[:half]**2)
            trend = recent_half_mse - older_half_mse  # positive = getting worse
        else:
            trend = 0.0
        
        # Base target from MSE
        target_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.15))
        
        # Adjust for trend: if errors are increasing, boost exploration more
        if trend > 0.01:
            target_alpha *= 1.2
        elif trend < -0.01:
            target_alpha *= 0.9
        
        # Smoothly move towards target with faster convergence
        self.alpha = 0.85 * self.alpha + 0.15 * target_alpha
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _update_reward_stats(self, reward):
        """Update running statistics of rewards for regime detection."""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) >= 5:
            rewards = np.array(self.recent_rewards)
            self.reward_mean = np.mean(rewards)
            self.reward_std = max(np.std(rewards), 0.1)

    def select(self, context):
        if not self._check_deadline():
            # If we are near the deadline, just pick the arm with highest mean prediction
            scores = []
            for a in range(self.n):
                try:
                    theta = self.inv_A[a] @ self.b[a]
                    mu = theta @ context
                except:
                    mu = 0.0
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.inv_A[a] @ self.b[a]
            mu = theta @ context
            var = context @ self.inv_A[a] @ context
            bonus = self.alpha * np.sqrt(var)
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        self.t += 1
        
        # 1. Calculate residual using the model *before* this update
        theta_old = self.inv_A[arm] @ self.b[arm]
        pred = theta_old @ context
        residual = reward - pred
        
        # Store residual
        self.error_windows[arm].append(residual)
        self.global_errors.append(residual)
        
        # Update reward statistics
        self._update_reward_stats(reward)
        
        # 2. Determine current forgetting factor
        lam = self._get_lambda(arm)
        
        # 3. Update matrices
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # Recompute inverse
        self._recompute_inv(arm)
        
        # 4. Update alpha
        self._update_alpha()
