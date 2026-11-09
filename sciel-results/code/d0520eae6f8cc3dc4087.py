"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Adaptive Discounted RLS with UCB for Non-Stationary Linear Contextual Bandits.
    
    This strategy uses Recursive Least Squares with an exponential forgetting factor 
    (lambda) to maintain a weighted least-squares fit of the most recent data. 
    Unlike a sliding window, this provides smooth adaptation. The forgetting factor 
    is adapted dynamically: if the prediction error is large, we assume the 
    environment has changed and decrease lambda (forget faster). If the error is 
    small, we increase lambda (remember more).
    
    Exploration is handled via Upper Confidence Bound (UCB), which adds a bonus 
    term proportional to the uncertainty (variance) of the reward estimate. This 
    is often more stable than Thompson Sampling in high-noise or non-stationary 
    settings.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Hyperparameters
        # Initial lambda close to 1 (remember everything), but will adapt.
        self.lambda_base = 0.95
        self.lambda_min = 0.5
        self.lambda_max = 0.99
        self.lambda_current = self.lambda_base
        
        # UCB exploration constant
        self.alpha_ucb = 1.0
        
        # Initialize sufficient statistics for RLS
        # We maintain P = (A)^-1 directly for O(d^2) updates.
        # A = sum lambda^t x x^T
        # P = A^-1
        # Initial P is large (high uncertainty). We use P0 = 100 * I.
        self.P = [100.0 * np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # For adaptive lambda, we track recent prediction errors per arm
        # to estimate the noise level and change rate.
        # We use a simple exponential moving average of squared errors.
        self.err_ema = [0.0 for _ in range(n_arms)]
        self.err_ema_alpha = 0.1 # smoothing factor for error tracking
        
        # Time management for safety
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            # Fallback: simple greedy based on current theta
            scores = [self.theta[a] @ np.asarray(context) for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            # Predicted mean
            mu = self.theta[a] @ context
            
            # Compute UCB bonus: alpha * sqrt(context^T P context)
            # P is the covariance matrix of theta (up to regularization factor)
            # In RLS, P = (lambda * A_prev + x x^T)^-1 ... wait, standard RLS update:
            # K = P x / (lambda + x^T P x)
            # theta = theta + K (y - x^T theta)
            # P = (P - K x^T) / lambda
            
            # The variance of the prediction y = x^T theta is x^T P x.
            # However, in our RLS formulation, P represents the inverse of the 
            # weighted sum of outer products. The actual variance of the estimate 
            # theta is P. The variance of the prediction x^T theta is x^T P x.
            
            try:
                # Compute variance of prediction
                var = context @ (self.P[a] @ context)
                # Ensure non-negative due to numerical errors
                var = max(var, 0.0)
                bonus = self.alpha_ucb * np.sqrt(var)
            except:
                bonus = 0.0

            scores.append(mu + bonus)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # 1. Calculate prediction error
        pred = self.theta[arm] @ context
        error = reward - pred
        error_sq = error * error
        
        # 2. Update adaptive lambda
        # If error is large, we want to forget faster (lower lambda).
        # If error is small, we can afford to remember more (higher lambda).
        # Base error level is assumed to be around 1.0 (standard deviation of noise).
        # We scale the adaptation rate based on the error magnitude.
        
        # Simple heuristic: 
        # lambda_new = lambda_min + (lambda_max - lambda_min) * exp(-error_sq / threshold)
        # Or even simpler: if error is high, drop lambda.
        
        # Update EMA of error
        self.err_ema[arm] = (1 - self.err_ema_alpha) * self.err_ema[arm] + self.err_ema_alpha * error_sq
        
        # Target lambda: 
        # If err_ema is low (< 1.0), we are in a stable regime, use high lambda (0.99).
        # If err_ema is high (> 4.0), we are in a changing regime, use low lambda (0.5).
        # Interpolate linearly or exponentially.
        
        # Let's use a smooth transition.
        # Map error variance [0, 10] to lambda [0.99, 0.5]
        # err_norm = min(self.err_ema[arm] / 10.0, 1.0)
        # lambda_target = 0.99 - 0.49 * err_norm
        
        # To avoid oscillation, we can just set lambda based on current error directly 
        # but smoothed.
        
        # A robust approach from literature: 
        # lambda_t = 1 - 1/(1 + c * error_t^2) ? No, that's for control.
        
        # Let's stick to the EMA-based target.
        err_norm = min(self.err_ema[arm] / 5.0, 1.0) # Normalize error
        lambda_target = self.lambda_max - (self.lambda_max - self.lambda_min) * err_norm
        
        # Smooth the lambda change itself to avoid abrupt shifts in dynamics
        # lambda_current = 0.9 * lambda_current + 0.1 * lambda_target
        # But since lambda is per-arm in theory, but often global in simple impls, 
        # let's keep it global for simplicity but derived from the active arm's error?
        # Actually, non-stationarity might be global. Let's update a global lambda 
        # based on the error of the selected arm, as that's the most relevant signal.
        
        # However, different arms might have different noise. 
        # Let's maintain a global lambda but update it based on the current arm's error.
        # This is a heuristic.
        
        # To be more precise, we can have per-arm lambda, but that complicates P updates 
        # if we switch arms. Standard RLS assumes constant lambda. 
        # Let's use a global lambda that adapts to the "worst" recent error or the 
        # current arm's error.
        
        # For this implementation, we will update a global self.lambda_current 
        # using the error from the current arm.
        self.lambda_current = 0.95 * self.lambda_current + 0.05 * lambda_target
        
        # Ensure bounds
        if self.lambda_current < self.lambda_min:
            self.lambda_current = self.lambda_min
        elif self.lambda_current > self.lambda_max:
            self.lambda_current = self.lambda_max
            
        lam = self.lambda_current

        # 3. RLS Update
        # K = P x / (lam + x^T P x)
        # theta = theta + K * error
        # P = (P - K x^T) / lam
        
        try:
            P = self.P[arm]
            Px = P @ context
            denom = lam + context @ Px
            
            if denom < 1e-6:
                # Numerical stability fallback
                denom = 1e-6
                
            K = Px / denom
            
            # Update theta
            self.theta[arm] += K * error
            
            # Update P
            # P_new = (P - K x^T) / lam
            # P - K x^T = P - (P x) x^T / denom
            # This can be computed as P @ (I - x x^T / denom) ? No.
            # P_new = (1/lam) * (P - (P x) (x^T) / denom)
            
            # To maintain symmetry and positive definiteness, we can use the 
            # Sherman-Morrison formula structure or just compute directly.
            # Direct computation:
            self.P[arm] = (P - np.outer(K, context)) / lam
            
            # Symmetrize P to prevent numerical drift
            self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
            
        except Exception:
            # If numerical issues arise, reset P to a large value (high uncertainty)
            # This effectively "forgets" everything and restarts learning for this arm.
            self.P[arm] = 100.0 * np.eye(self.d)
            self.theta[arm] = np.zeros(self.d)

        self.t += 1
# EVOLVE-BLOCK-END
