"""Non-stationary linear contextual bandit with global arm-augmented model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Global linear bandit with arm-augmented context and adaptive forgetting.
    
    The context is augmented with one-hot arm indicators to create a single
    global model. Adaptive forgetting based on prediction error allows tracking
    of non-stationary changes. Sherman-Morrison-like updates for efficiency.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms  # augmented dimension
        
        # Global model: A is inverse covariance, b is weighted sum of contexts*rewards
        self.A = np.eye(self.d_aug)
        self.b = np.zeros(self.d_aug)
        
        # Adaptive forgetting parameters
        self.forgetting_base = 0.99
        self.forgetting_min = 0.95
        self.forgetting_max = 0.999
        self.error_window = []
        self.error_window_size = 50
        
        # Exploration parameter
        self.alpha = 1.0
        self.alpha_decay = 0.9999
        self.alpha_min = 0.5
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # seconds, safety margin before 150s
        
        # Track best arm for fallback
        self.best_arm = 0
        self.best_reward = -float('inf')
        self.reward_history = []

    def _augment_context(self, context, arm):
        """Augment context with one-hot arm indicator."""
        aug = np.zeros(self.d_aug)
        aug[:self.d] = context
        aug[self.d + arm] = 1.0
        return aug

    def _select_arm(self, context):
        """Select arm using UCB with global model."""
        scores = []
        for a in range(self.n):
            aug_ctx = self._augment_context(context, a)
            # Predicted reward
            theta = np.linalg.solve(self.A, self.b)
            pred = theta @ aug_ctx
            # Uncertainty bonus
            # Use diagonal of A for approximation (faster than full inverse)
            variance = aug_ctx @ self.A @ aug_ctx
            bonus = self.alpha * np.sqrt(max(variance, 0))
            scores.append(pred + bonus)
        
        return int(np.argmax(scores))

    def _adaptive_forgetting(self, prediction_error):
        """Compute adaptive forgetting factor based on recent prediction errors."""
        self.error_window.append(abs(prediction_error))
        if len(self.error_window) > self.error_window_size:
            self.error_window.pop(0)
        
        if len(self.error_window) < 5:
            return self.forgetting_base
        
        mean_error = np.mean(self.error_window)
        
        # Map error to forgetting factor: higher error -> lower forgetting (faster adaptation)
        # Normalize error to [0, 1] range
        error_scale = 2.0  # scale factor for normalization
        normalized_error = min(mean_error / error_scale, 1.0)
        
        # Linearly interpolate between forgetting_min and forgetting_max
        forgetting = self.forgetting_max - normalized_error * (self.forgetting_max - self.forgetting_min)
        return max(self.forgetting_min, min(self.forgetting_max, forgetting))

    def select(self, context):
        """Select arm based on context."""
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return self.best_arm
        
        return self._select_arm(context)

    def update(self, context, arm, reward):
        """Update model with observed reward."""
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return
        
        # Track best arm
        self.reward_history.append(reward)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = arm
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        
        # Augment context
        aug_ctx = self._augment_context(context, arm)
        
        # Compute prediction before update for adaptive forgetting
        theta = np.linalg.solve(self.A, self.b)
        pred = theta @ aug_ctx
        prediction_error = reward - pred
        
        # Adaptive forgetting
        forgetting = self._adaptive_forgetting(prediction_error)
        
        # Update with forgetting
        self.A = forgetting * self.A
        self.b = forgetting * self.b
        
        # Sherman-Morrison update: A_new = A + outer(ctx, ctx)
        self.A += np.outer(aug_ctx, aug_ctx)
        self.b += reward * aug_ctx
        
        # Decay exploration parameter
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
# EVOLVE-BLOCK-END
