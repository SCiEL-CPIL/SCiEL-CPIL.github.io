"""
Non-stationary linear contextual bandit policy.
Uses a single global linear model with arm-specific bias terms and
adaptive exponential decay for non-stationary tracking.
"""

import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global linear model: A (precision) and b (moment vector)
        self.A = np.eye(dim)
        self.b = np.zeros(dim)
        
        # Arm-specific bias terms (scalar per arm)
        self.arm_bias = np.zeros(n_arms)
        self.arm_bias_counts = np.zeros(n_arms)
        
        # Adaptive forgetting parameters
        self.base_lambda = 0.98
        self.min_lambda = 0.80   # Faster forgetting for regime changes
        self.max_lambda = 0.995  # Slow forgetting when stable
        
        # Error tracking (global, not per-arm)
        self.error_buffer = []
        self.error_window_size = 30
        
        # UCB exploration parameter
        self.alpha = 2.0
        
        # Track last predictions for error computation
        self._last_pred = 0.0
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0
        
        # Track rounds
        self.total_rounds = 0
        
        # Global reward statistics
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0

    def _get_lambda(self):
        """
        Calculate the adaptive forgetting factor based on recent
        global prediction errors.
        """
        if not self.error_buffer:
            return self.base_lambda
            
        recent = self.error_buffer[-self.error_window_size:]
        mean_err = np.mean(recent)
        
        # Normalize error with a soft cap
        err_scale = 0.15
        normalized_err = mean_err / err_scale
        
        # Smooth mapping: high error -> low lambda (forget fast)
        if normalized_err <= 0.05:
            return self.max_lambda
        elif normalized_err >= 2.0:
            return self.min_lambda
        else:
            t = (normalized_err - 0.05) / (2.0 - 0.05)
            t = t * t * (3 - 2 * t)
            return self.max_lambda - t * (self.max_lambda - self.min_lambda)

    def _get_alpha(self):
        """
        Adaptively adjust the exploration parameter based on global performance.
        """
        if self.total_rounds < 50:
            return self.alpha
        
        mean_reward = self.reward_sum / self.total_rounds
        var_reward = max(0, self.reward_sq_sum / self.total_rounds - mean_reward**2)
        std_reward = np.sqrt(var_reward)
        
        base_alpha = 1.5
        if std_reward > 0.3:
            alpha = base_alpha + 1.5
        elif std_reward > 0.1:
            alpha = base_alpha + 0.5
        else:
            alpha = base_alpha - 0.5
        
        return max(0.5, min(4.0, alpha))

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))
        
        context = np.asarray(context)
        alpha = self._get_alpha()
        
        try:
            inv_A = np.linalg.inv(self.A)
            theta = inv_A @ self.b
            
            # Mean prediction from global model
            mu_global = theta @ context
            
            # Variance term (shared across arms)
            var = context @ inv_A @ context
            std = np.sqrt(max(0, var))
            
            # Score each arm: global prediction + arm bias + exploration
            scores = []
            for a in range(self.n):
                # Arm-specific bias (initialized to 0, updated from rewards)
                bias = self.arm_bias[a] if self.arm_bias_counts[a] > 0 else 0.0
                score = mu_global + bias + alpha * std
                scores.append(score)
        except np.linalg.LinAlgError:
            scores = [0.0] * self.n
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # Deadline check
        if time.time() > self._deadline:
            return
        
        # Update global reward statistics
        self.total_rounds += 1
        self.reward_sum += reward
        self.reward_sq_sum += reward * reward
        
        # 1. Compute prediction error for adaptive lambda
        # Estimate what the global model would have predicted
        try:
            inv_A = np.linalg.inv(self.A)
            theta = inv_A @ self.b
            pred_global = theta @ context
        except np.linalg.LinAlgError:
            pred_global = 0.0
        
        # Include arm bias in prediction
        bias = self.arm_bias[arm] if self.arm_bias_counts[arm] > 0 else 0.0
        pred = pred_global + bias
        error = abs(reward - pred)
        
        # Append error to global buffer
        self.error_buffer.append(error)
        if len(self.error_buffer) > self.error_window_size * 2:
            self.error_buffer = self.error_buffer[-self.error_window_size:]
        
        # 2. Update global model with adaptive forgetting
        lam = self._get_lambda()
        
        self.A = lam * self.A + np.outer(context, context)
        self.b = lam * self.b + reward * context
        
        # 3. Update arm-specific bias
        # The bias captures the arm-specific offset from the global model
        # We want: reward ≈ theta @ context + bias_arm
        # So bias_arm should track (reward - theta @ context)
        # Use exponential moving average for the bias
        bias_lam = 0.9  # Slower decay for bias than for global model
        if self.arm_bias_counts[arm] == 0:
            self.arm_bias[arm] = reward - pred_global
        else:
            self.arm_bias[arm] = bias_lam * self.arm_bias[arm] + (1 - bias_lam) * (reward - pred_global)
        self.arm_bias_counts[arm] += 1
        
        # 4. Periodic regularization to prevent numerical issues
        if self.total_rounds % 100 == 0:
            self.A += 0.001 * np.eye(self.d)
