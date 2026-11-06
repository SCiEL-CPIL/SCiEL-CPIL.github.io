"""Non-stationary linear contextual bandit with global model + arm biases."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Global linear model with arm-specific bias terms.
    
    Instead of maintaining separate full linear models per arm, we maintain:
    - A single global linear model (A_global, b_global) that captures shared context features
    - Per-arm bias terms (bias[arm]) that capture arm-specific offsets
    
    The predicted reward for arm a given context x is:
        theta_global @ x + bias[a]
    
    This reduces parameter count, enables immediate exploitation of shared features,
    and allows cross-arm correlation to improve sample efficiency in non-stationary settings.
    
    We use a forgetting factor to handle non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global linear model
        self.A = np.eye(dim)
        self.b = np.zeros(dim)
        
        # Arm-specific biases
        self.bias = np.zeros(n_arms)
        self.bias_count = np.zeros(n_arms)
        
        # Forgetting factor
        self.alpha = 0.99
        
        # Exploration parameters
        self.beta = 1.0
        self.bias_beta = 1.0
        
        # Track total updates for adaptive beta
        self.total_updates = 0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute global prediction and uncertainty
        inv_A = np.linalg.inv(self.A)
        theta = inv_A @ self.b
        global_pred = theta @ context
        global_var = context @ inv_A @ context
        
        # Adaptive exploration: increase exploration early, decrease over time
        # but keep some minimum exploration for non-stationarity
        base_exploration = self.beta * np.sqrt(global_var)
        
        scores = []
        for a in range(self.n):
            # Predicted reward = global linear prediction + arm bias
            pred = global_pred + self.bias[a]
            
            # Uncertainty: combine global model uncertainty with bias uncertainty
            # Bias uncertainty is higher when we've seen fewer samples for this arm
            bias_var = self.bias_beta / (1.0 + self.bias_count[a])
            total_var = global_var + bias_var
            exploration = np.sqrt(total_var)
            
            scores.append(pred + self.beta * exploration)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        self.total_updates += 1
        
        # Decay the global model (forgetting factor for non-stationarity)
        self.A = self.alpha * self.A + np.outer(context, context)
        self.b = self.alpha * self.b + reward * context
        
        # Update arm-specific bias
        # We model the residual: reward - (global prediction)
        inv_A = np.linalg.inv(self.A)
        theta = inv_A @ self.b
        global_pred = theta @ context
        residual = reward - global_pred
        
        # Update bias with forgetting
        self.bias[arm] = self.alpha * self.bias[arm] + residual
        self.bias_count[arm] = self.alpha * self.bias_count[arm] + 1.0
        
        # Adapt beta: start with higher exploration, decay over time
        # but maintain a floor for non-stationary adaptation
        # beta = max(1.0, 2.0 / np.sqrt(self.total_updates + 1))
        # Keep beta fixed at 1.0 for simplicity and stability
# EVOLVE-BLOCK-END
