# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Global linear model with arm-augmented context for non-stationary bandits.
    
    Uses a single linear model over an augmented feature vector:
    [one_hot(arm), context]
    
    This captures arm-specific offsets and shared context effects in a unified
    parameter space, reducing total parameters compared to per-arm models
    (n * d + d vs n * d) and allowing cross-arm information sharing.
    
    Uses exponential forgetting to adapt to non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = n_arms + dim  # augmented dimension
        
        # Global model
        self.A = np.eye(self.d_aug)
        self.b = np.zeros(self.d_aug)
        
        # Forgetting factor for non-stationarity
        self.forgetting = 0.99
        
        # Exploration parameter
        self.alpha = 1.0
        
        # Track last arm for potential adjustments
        self.last_arm = None

    def _augment(self, arm, context):
        """Create augmented feature vector: [one_hot(arm), context]."""
        one_hot = np.zeros(self.n)
        one_hot[arm] = 1.0
        return np.concatenate([one_hot, context])

    def select(self, context):
        """Select arm using optimistic estimate from global model."""
        scores = []
        inv_A = np.linalg.inv(self.A)
        theta = inv_A @ self.b
        
        for a in range(self.n):
            x_aug = self._augment(a, context)
            # Predicted reward
            pred = theta @ x_aug
            # Uncertainty bonus
            var = x_aug @ inv_A @ x_aug
            bonus = self.alpha * np.sqrt(var)
            scores.append(pred + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update global model with observed (context, arm, reward)."""
        x_aug = self._augment(arm, context)
        
        # Apply forgetting
        self.A = self.forgetting * self.A + np.outer(x_aug, x_aug)
        self.b = self.forgetting * self.b + reward * x_aug
# EVOLVE-BLOCK-END
