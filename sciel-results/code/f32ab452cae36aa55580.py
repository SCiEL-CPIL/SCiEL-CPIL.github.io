"""
Non-stationary Linear Contextual Bandit with Global Surprise-Triggered Reset.

Strategy:
1. Use a fixed, moderately fast forgetting factor (gamma=0.99) for all arms.
2. Maintain a global "surprise" detector based on the standardized residual of the last observation.
3. If the surprise exceeds a threshold, perform a "soft reset" on the model for the pulled arm:
   - Reset A_inv to a small multiple of Identity (reducing confidence).
   - Reset b to 0.
   - This effectively clears the memory of the old regime for that arm immediately.
4. This avoids the lag and instability of adaptive gamma tuning and responds sharply to non-stationarity.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin

        # Hyperparameters
        self.alpha = 1.2       # Exploration bonus
        self.reg = 1.0         # Initial regularization
        self.gamma = 0.99      # Fixed forgetting factor
        
        # Reset parameters
        self.reset_threshold = 3.0  # Z-score threshold for surprise
        self.reset_reg = 0.1       # Regularization after reset (low confidence)
        
        # Initialize models for each arm
        # A_inv: Inverse of the regularized covariance matrix
        # b: Regularized sum of x*y
        self.A_inv = [np.eye(self.d) / self.reg for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]
        
        # Track recent residuals for global detection
        # We only need the last one for the "surprise" check, but keep a small history
        # to estimate the baseline noise level if needed, though for simplicity we use
        # the model's own uncertainty.
        self.last_residual = 0.0
        self.round = 0

    def _get_value_and_uncertainty(self, arm, context):
        """
        Compute the estimated mean and uncertainty for an arm.
        """
        x = context
        A_inv = self.A_inv[arm]
        b = self.b[arm]

        # Mean estimate
        theta = A_inv @ b
        mean_val = theta @ x
        
        # Uncertainty
        var = x @ A_inv @ x
        std = np.sqrt(var)
        
        return mean_val, std

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return int(np.random.randint(self.n))
        
        scores = []
        for a in range(self.n):
            mean_val, uncertainty = self._get_value_and_uncertainty(a, context)
            # UCB-style score
            score = mean_val + self.alpha * uncertainty
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x = context
        y = reward
        
        # 1. Compute the residual (surprise) BEFORE updating the model
        # This tells us if the new observation was unexpected given the current model.
        mean_val, std = self._get_value_and_uncertainty(arm, x)
        if std > 1e-8:
            z_score = (y - mean_val) / std
        else:
            z_score = 0.0
            
        self.last_residual = y - mean_val
        
        # 2. Check for non-stationarity (Surprise)
        # If the residual is large relative to the model's confidence, 
        # the model is likely stale. Reset the model for this arm.
        if abs(z_score) > self.reset_threshold:
            # Soft Reset: Reduce confidence (increase A_inv) and clear bias
            # This allows the next few updates to quickly re-learn the new weights
            self.A_inv[arm] = np.eye(self.d) / self.reset_reg
            self.b[arm] = np.zeros(self.d)
            # After reset, the "surprise" is effectively handled. 
            # We proceed to update with the new data point.
            
            # Update the reset model with the current observation
            # Since we just reset, the update is simple:
            # A_inv_new = (1/gamma) * A_inv - (1/gamma) * (A_inv x x^T A_inv) / (gamma + x^T A_inv x)
            # But since we want to incorporate this point immediately into the fresh model:
            A_inv = self.A_inv[arm]
            b = self.b[arm]
            
            Ax = A_inv @ x
            denom = self.gamma + x @ Ax
            A_inv_new = (1.0 / self.gamma) * A_inv
            correction = np.outer(Ax, Ax) / denom
            A_inv_new -= (1.0 / self.gamma) * correction
            A_inv_new = (A_inv_new + A_inv_new.T) / 2.0
            
            b_new = self.gamma * b + y * x
            
            self.A_inv[arm] = A_inv_new
            self.b[arm] = b_new
        else:
            # 3. Normal Update with Forgetting
            A_inv = self.A_inv[arm]
            b = self.b[arm]
            
            Ax = A_inv @ x
            denom = self.gamma + x @ Ax
            A_inv_new = (1.0 / self.gamma) * A_inv
            correction = np.outer(Ax, Ax) / denom
            A_inv_new -= (1.0 / self.gamma) * correction
            A_inv_new = (A_inv_new + A_inv_new.T) / 2.0
            
            b_new = self.gamma * b + y * x
            
            self.A_inv[arm] = A_inv_new
            self.b[arm] = b_new
            
        self.round += 1

# EVOLVE-BLOCK-END
