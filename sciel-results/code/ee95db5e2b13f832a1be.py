import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Model parameters for each arm
        # A: d x d matrix, b: d vector
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Per-arm adaptive forgetting factors
        # lambda in [lambda_min, lambda_max]
        self.lambda_min = 0.85
        self.lambda_max = 0.999
        self.lambdas = [0.99 for _ in range(n_arms)]
        
        # Error tracking for adaptive lambda
        # We track the exponential moving average of squared errors per arm
        self.err_ema = [0.0 for _ in range(n_arms)]
        self.err_alpha = 0.1 # Learning rate for error EMA
        
        # Exploration parameter
        self.alpha_start = 2.0
        self.alpha_min = 0.5
        self.decay_rate = 0.001
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0 # 5s margin
        
        # Reset threshold multiplier
        self.reset_threshold = 3.0

    def _solve(self, A, b):
        """Solve linear system with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _get_inverse_solve(self, A, vec):
        """Solve A x = vec."""
        try:
            return np.linalg.solve(A, vec)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, vec, rcond=None)[0]

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n_arms):
                theta = self._solve(self.A[a], self.b[a])
                mu = theta @ context
                if mu > best_score:
                    best_score = mu
                    best_arm = a
            return best_arm

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Adaptive Alpha
        alpha = max(self.alpha_min, self.alpha_start * np.exp(-self.decay_rate * self.t))
        
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            A = self.A[a]
            b = self.b[a]
            
            theta = self._solve(A, b)
            mu = theta @ context
            
            # Uncertainty term: x^T A^{-1} x
            A_inv_x = self._get_inverse_solve(A, context)
            v = context @ A_inv_x
            
            # UCB Score
            score = mu + alpha * np.sqrt(max(v, 0.0))
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Compute prediction and residual
        A = self.A[arm]
        b = self.b[arm]
        theta = self._solve(A, b)
        mu_pred = theta @ context
        
        # Uncertainty for reset check
        A_inv_x = self._get_inverse_solve(A, context)
        v = context @ A_inv_x
        std_pred = np.sqrt(max(v, 0.0)) + 1.0 # Add baseline noise
        
        residual = reward - mu_pred
        sq_error = residual ** 2
        
        # 2. Update Per-Arm Lambda
        # Update Error EMA
        self.err_ema[arm] = (1 - self.err_alpha) * self.err_ema[arm] + self.err_alpha * sq_error
        
        # Map Error EMA to Lambda
        # High error -> Low lambda (fast forgetting)
        # Low error -> High lambda (slow forgetting)
        # We want lambda to drop significantly if error is large.
        # Let's use a soft min-like function or a simple linear clamp.
        # Assume typical variance is ~1.0.
        # If err_ema > 2, we want lambda close to lambda_min.
        # If err_ema < 0.5, we want lambda close to lambda_max.
        
        # Normalize error
        err_norm = self.err_ema[arm] / (self.err_ema[arm] + 1.0)
        
        # Map [0, 1] to [lambda_max, lambda_min]
        # lambda = lambda_max - (lambda_max - lambda_min) * err_norm
        new_lambda = self.lambda_max - (self.lambda_max - self.lambda_min) * err_norm
        
        # Smooth the lambda update to avoid oscillation
        self.lambdas[arm] = 0.8 * self.lambdas[arm] + 0.2 * new_lambda
        
        lam = self.lambdas[arm]
        
        # 3. Check for Shock Reset
        # If residual is significantly larger than expected standard deviation
        if abs(residual) > self.reset_threshold * std_pred:
            # Reset model for this arm to prioritize the new data point
            # Also reset error EMA to avoid immediate re-adaptation
            self.A[arm] = np.eye(self.dim) * 1.0
            self.b[arm] = reward * context
            self.err_ema[arm] = 0.0
            self.lambdas[arm] = self.lambda_min # Start with fast forgetting after shock
        else:
            # Standard RLS update with per-arm forgetting
            self.A[arm] = lam * self.A[arm] + np.outer(context, context)
            self.b[arm] = lam * self.b[arm] + reward * context
