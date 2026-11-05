# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit using Adaptive RLS per arm.
    
    Improvements over seed:
    - Replaces fixed forgetting factor with adaptive forgetting based on prediction error.
    - Uses Recursive Least Squares (RLS) update for faster convergence and better tracking.
    - Per-arm models to handle distinct reward structures.
    - UCB exploration based on covariance.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize per-arm parameters
        # theta: weight vector
        # P: inverse covariance matrix (used for RLS update and UCB)
        self.thetas = [np.zeros(dim) for _ in range(n_arms)]
        self.Ps = [np.eye(dim) * 10.0 for _ in range(n_arms)] # High initial uncertainty
        
        # RLS parameters
        self.lambda_min = 0.95  # Minimum forgetting factor (slowest forgetting)
        self.lambda_max = 0.999 # Maximum forgetting factor (fastest forgetting)
        self.alpha = 1.0        # Learning rate scale for RLS
        
        # Exploration
        self.gamma = 1.0        # UCB exploration constant
        
        # Adaptive forgetting parameters
        self.error_threshold = 1.0
        self.adaptation_speed = 0.5
        
        # Time tracking for safety
        self.start_time = time.time()
        self.elapsed = 0.0

    def _adaptive_lambda(self, arm, context, reward):
        """
        Calculate an adaptive forgetting factor based on the prediction error.
        Large errors (regime shifts) lead to smaller lambda (faster forgetting).
        """
        # Predicted reward
        pred = np.dot(self.thetas[arm], context)
        error = reward - pred
        
        # Normalize error to determine how "surprising" it is
        # We use a simple heuristic: if error is large, we forget more
        # We map error magnitude to a range [lambda_max, lambda_min]
        # Using a sigmoid-like or linear mapping
        # Let's use a linear mapping clipped
        error_mag = np.abs(error)
        
        # Scale error to a factor between 0 and 1
        # If error is 0, factor is 0 (use lambda_max, slow forgetting)
        # If error is large, factor is 1 (use lambda_min, fast forgetting)
        # We use a soft cap to avoid extreme jumps
        factor = np.tanh(error_mag / self.error_threshold)
        
        # Interpolate between lambda_max and lambda_min
        # Higher factor -> lower lambda (faster forgetting)
        lam = self.lambda_max - factor * (self.lambda_max - self.lambda_min)
        
        return lam

    def select(self, context):
        # Check time budget
        self.elapsed = time.time() - self.start_time
        if self.elapsed > 110.0:
             # Fallback to greedy if time is running out
             scores = [np.dot(self.thetas[a], context) for a in range(self.n)]
             return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Mean reward
            mean = np.dot(self.thetas[a], context)
            
            # Variance estimate for UCB
            # var = gamma^2 * context^T P context
            var = np.dot(context, self.Ps[a].dot(context))
            bonus = self.gamma * np.sqrt(var)
            
            scores.append(mean + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update the model for the selected arm
        # 1. Calculate adaptive forgetting factor
        lam = self._adaptive_lambda(arm, context, reward)
        
        # 2. RLS Update
        # P_new = (1/lam) * (P_old - (P_old x (x^T P_old x)^-1 x^T P_old))
        # theta_new = theta_old + P_new x (y - x^T theta_old)
        
        P = self.Ps[arm]
        theta = self.thetas[arm]
        
        # Prediction error
        pred = np.dot(theta, context)
        error = reward - pred
        
        # Calculate gain vector
        # K = P x / (lam + x^T P x)
        Px = P.dot(context)
        denom = lam + np.dot(context, Px)
        
        # Numerical stability check
        if denom < 1e-6:
            denom = 1e-6
            
        K = Px / denom
        
        # Update theta
        theta_new = theta + K * error
        self.thetas[arm] = theta_new
        
        # Update P
        # P_new = (P - K x^T P) / lam
        P_new = (P - np.outer(K, context).dot(P)) / lam
        
        # Symmetrize to prevent numerical drift
        P_new = (P_new + P_new.T) / 2.0
        
        self.Ps[arm] = P_new

# EVOLVE-BLOCK-END
