# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Non-stationary linear contextual bandit using Recursive Least Squares (RLS)
    with a forgetting factor for each arm.
    
    This approach maintains the weight vector theta and the inverse covariance matrix P
    for each arm. The RLS update rule allows for efficient O(d^2) updates per observation
    and naturally incorporates a forgetting factor to track non-stationary dynamics.
    
    Exploration is done via UCB using the uncertainty derived from the covariance matrix P.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize weights to zero
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize inverse covariance matrices to large values (high uncertainty)
        # Using a large scalar multiple of identity ensures stability initially
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        # Forgetting factor: 0.98 means we keep 98% of old information, discard 2%
        self.lambda_ = 0.98
        # Exploration parameter
        self.alpha = 1.5
        self.t = 0
        self.visits = [0] * n_arms

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Mean reward estimate
            mean = self.theta[a] @ context
            
            # Uncertainty term: alpha * sqrt(context^T P context)
            # P is the inverse covariance, so context^T P context is the variance of the estimate
            variance = context @ (self.P[a] @ context)
            
            # Adaptive alpha: slightly higher for less visited arms
            # Simple heuristic: decay alpha with time, floor at 0.5
            current_alpha = max(0.5, self.alpha / (1.0 + self.t * 0.001))
            
            uncertainty = current_alpha * np.sqrt(max(0.0, variance))
            
            scores.append(mean + uncertainty)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # RLS Update Equations with Forgetting Factor
        # 1. Compute gain K
        # K = P * x / (lambda + x^T * P * x)
        P_x = self.P[arm] @ context
        denom = self.lambda_ + context @ P_x
        
        if denom < 1e-8:
            return # Avoid division by zero or extremely small numbers
            
        K = P_x / denom
        
        # 2. Update theta
        # theta = theta + K * (reward - x^T * theta)
        prediction_error = reward - (self.theta[arm] @ context)
        self.theta[arm] = self.theta[arm] + K * prediction_error
        
        # 3. Update P
        # P = (P - K * x^T * P) / lambda
        # Efficient form: P = (1/lambda) * (P - K * x^T * P)
        # Note: K * x^T is a rank-1 matrix
        self.P[arm] = (self.P[arm] - np.outer(K, context) @ self.P[arm]) / self.lambda_
        
        # Symmetrize P to prevent numerical drift
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
        
        self.visits[arm] += 1
# EVOLVE-BLOCK-END
