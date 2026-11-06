"""Non-stationary linear contextual bandit with global model + arm biases and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear bandit with a global linear model and arm-specific biases.
    
    Structure: reward = theta_global @ context + bias[arm] + noise
    This shares context-feature weights across arms, improving sample efficiency
    while maintaining arm-specific adaptation through bias terms.
    
    Adaptive forgetting factor and exploration bonus respond to recent prediction errors.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._start_time = time.time()
        self._deadline = 115.0  # safety margin before 120s limit
        
        # Global linear model (shared across arms)
        self.A_global = np.eye(dim) * 1.0  # Regularized covariance
        self.b_global = np.zeros(dim)
        
        # Arm-specific biases
        self.bias = np.zeros(n_arms)
        self.bias_counts = np.ones(n_arms)  # For running mean of bias
        
        # Per-arm bias tracking with forgetting
        self.bias_A = np.ones(n_arms)  # Effective sample count for bias
        self.bias_b = np.zeros(n_arms)  # Sum of (reward - global_pred) * context_norm_factor
        
        # Adaptive parameters
        self.base_alpha = 1.0
        self.base_forgetting = 0.95
        
        # Track recent prediction errors for adaptivity
        self.recent_errors = []
        self.error_window = 20
        self.error_threshold_high = 1.0
        self.error_threshold_low = 0.1
        
        # Round counter
        self.round = 0
        
        # Per-arm visit counts for bias uncertainty
        self.arm_visits = np.zeros(n_arms)

    def _get_adaptive_params(self):
        """Compute adaptive forgetting factor and exploration coefficient."""
        if len(self.recent_errors) < 5:
            return self.base_forgetting, self.base_alpha
        
        recent = self.recent_errors[-self.error_window:]
        mean_err = np.mean(np.abs(recent))
        std_err = np.std(recent)
        
        # Adaptive forgetting: decrease (more forgetting) when errors are high
        if mean_err > self.error_threshold_high:
            excess = min(mean_err / self.error_threshold_high, 5.0)
            forgetting = self.base_forgetting - 0.1 * min(excess, 1.0)
            forgetting = max(0.80, min(0.99, forgetting))
        elif mean_err < self.error_threshold_low:
            ratio = mean_err / self.error_threshold_low
            forgetting = min(0.995, self.base_forgetting + 0.04 * (1.0 - ratio))
        else:
            t = (mean_err - self.error_threshold_low) / (self.error_threshold_high - self.error_threshold_low)
            forgetting = self.base_forgetting - 0.05 * t
        
        # Adaptive alpha: increase exploration when errors are high
        if mean_err > self.error_threshold_high:
            excess = min(mean_err / self.error_threshold_high, 3.0)
            alpha = self.base_alpha + 1.0 * min(excess, 2.0)
        elif mean_err < self.error_threshold_low:
            ratio = mean_err / self.error_threshold_low
            alpha = max(0.3, self.base_alpha - 0.5 * (1.0 - ratio))
        else:
            t = (mean_err - self.error_threshold_low) / (self.error_threshold_high - self.error_threshold_low)
            alpha = self.base_alpha + 0.5 * t
        
        # Add small component from std to capture volatility
        alpha += 0.2 * min(std_err / max(mean_err, 1e-8), 1.0)
        
        return forgetting, alpha

    def _get_global_theta(self):
        """Solve for global theta vector."""
        try:
            theta = np.linalg.solve(self.A_global, self.b_global)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(self.A_global) @ self.b_global
        return theta

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Check deadline
        if time.time() - self._start_time > self._deadline:
            theta = self._get_global_theta()
            scores = self.bias + theta @ context
            return int(np.argmax(scores))
        
        _, alpha = self._get_adaptive_params()
        
        theta = self._get_global_theta()
        global_mean = float(theta @ context)
        
        # Uncertainty from global model
        try:
            inv_A_ctx = np.linalg.solve(self.A_global, context)
            global_var = float(context @ inv_A_ctx)
        except np.linalg.LinAlgError:
            inv_A = np.linalg.pinv(self.A_global)
            global_var = float(context @ inv_A @ context)
        
        global_bonus = alpha * np.sqrt(max(global_var, 0.0))
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            mean = global_mean + self.bias[a]
            
            # Bias uncertainty: larger for rarely visited arms
            visits = self.arm_visits[a]
            if visits < 1:
                bias_var = 1.0
            else:
                bias_var = 1.0 / max(visits, 1.0)
            
            bonus = global_bonus + 0.5 * np.sqrt(bias_var)
            scores[a] = mean + bonus
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.round += 1
        
        # Compute prediction error before update
        theta = self._get_global_theta()
        pred = float(theta @ context) + self.bias[arm]
        error = reward - pred
        
        # Track recent errors
        self.recent_errors.append(abs(error))
        if len(self.recent_errors) > self.error_window * 2:
            self.recent_errors = self.recent_errors[-self.error_window * 2:]
        
        # Update with adaptive forgetting
        forgetting, _ = self._get_adaptive_params()
        
        # Update global model
        self.A_global = forgetting * self.A_global + np.outer(context, context)
        self.b_global = forgetting * self.b_global + reward * context
        
        # Update arm-specific bias
        # Residual after removing global prediction
        residual = reward - float(theta @ context)
        self.bias_b[arm] = forgetting * self.bias_b[arm] + residual
        self.bias_A[arm] = forgetting * self.bias_A[arm] + 1.0
        self.bias[arm] = self.bias_b[arm] / max(self.bias_A[arm], 1e-6)
        
        # Track visits
        self.arm_visits[arm] = forgetting * self.arm_visits[arm] + 1.0
        
        # Periodic re-regularization to prevent numerical drift
        if self.round % 100 == 0:
            self.A_global += 1e-4 * np.eye(self.d)
# EVOLVE-BLOCK-END
