"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """Adaptive LinUCB for non-stationary linear contextual bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        
        # Per-arm linear model parameters
        # A is the inverse covariance matrix, b is the linear term
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor parameters
        self.alpha = 0.99  # Current forgetting factor
        self.min_alpha = 0.95
        self.max_alpha = 0.999
        self.alpha_target = 0.99
        
        # Error tracking for adaptivity
        self.error_window_size = 20
        self.recent_errors = []
        
        # Exploration coefficient
        self.c = 1.0

    def select(self, context):
        # Deadline guard: if we are close to the limit, return a safe arm (e.g., arm 0 or best known)
        # In this simple loop, the computation is very fast, so this is mostly a safety net.
        # However, we must ensure we don't hang.
        
        scores = []
        for a in range(self.n):
            # Compute theta = A^{-1} b
            # Since A is maintained as the inverse, we just multiply
            theta = self.A[a] @ self.b[a]
            
            # Mean estimate
            mu = theta @ context
            
            # Variance estimate: context^T A context
            # Note: In standard LinUCB, A is the inverse of (sum xx^T + lambda I).
            # So var = x^T A x.
            var = context @ (self.A[a] @ context)
            
            # UCB score
            # Use sqrt(var) for exploration
            ucb = mu + self.c * np.sqrt(max(var, 0))
            scores.append(ucb)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Update adaptive alpha based on recent errors
        # Calculate prediction error for the selected arm
        # We need the predicted value before update to calculate error
        # But update is called after select. We can store the last prediction.
        # Alternatively, just use the reward vs current estimate.
        
        # For simplicity and robustness, we adapt alpha based on the magnitude of the reward deviation 
        # from the previous model's prediction.
        
        # Get previous prediction for this arm
        theta = self.A[arm] @ self.b[arm]
        pred = theta @ context
        error = reward - pred
        
        # Track recent errors
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window_size:
            self.recent_errors.pop(0)
            
        # Calculate average absolute error
        if self.recent_errors:
            avg_err = np.mean(np.abs(self.recent_errors))
            # Normalize error to determine target alpha
            # If error is high, we want to forget faster (lower alpha)
            # If error is low, we want to forget slower (higher alpha)
            # Map avg_err to [min_alpha, max_alpha]
            # Assume typical error scale is around 1.0
            # If avg_err > 1.0, push alpha down.
            # If avg_err < 0.5, push alpha up.
            
            # Simple linear mapping with clamping
            # target = max_alpha - k * avg_err
            # Let's use a smooth transition
            self.alpha_target = self.max_alpha - 0.1 * avg_err
            self.alpha_target = np.clip(self.alpha_target, self.min_alpha, self.max_alpha)
            
            # Smoothly update alpha
            self.alpha = 0.9 * self.alpha + 0.1 * self.alpha_target
        else:
            self.alpha = 0.99
            
        # Update model with forgetting factor
        # A_new = alpha * A_old + x x^T
        # b_new = alpha * b_old + r x
        # Note: This formulation assumes A is the inverse of the regularized covariance.
        # Standard LinUCB update:
        # A <- A + x x^T
        # b <- b + r x
        # With forgetting:
        # A <- alpha * A + x x^T
        # b <- alpha * b + r x
        # This is equivalent to re-weighting past observations exponentially.
        
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context

# EVOLVE-BLOCK-END
