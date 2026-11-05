# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Bandit using Thompson Sampling with adaptive covariance inflation.
    
    This policy maintains a Gaussian posterior belief for each arm's weight vector.
    It uses an adaptive forgetting mechanism where the forgetting factor (gamma) 
    is determined by the standardized innovation (residual) magnitude. Large residuals 
    trigger more aggressive forgetting (covariance inflation) to track non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lambda_reg = 1.0  # Prior precision (inverse variance)
        
        # Initialize precision matrices and vectors
        self.A = [self.lambda_reg * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Timing
        self.t0 = time.time()
        self.deadline = 145.0  # Safety margin under 150s kill
        
        # Adaptive forgetting parameters
        # gamma = exp(-beta * z^2)
        # If z is small (stable), gamma ~ 1 (little forgetting)
        # If z is large (drift), gamma < 1 (more forgetting)
        self.beta_forget = 0.15 
        self.gamma_min = 0.5
        self.gamma_max = 0.99

    def _check_deadline(self):
        return (time.time() - self.t0) < self.deadline

    def select(self, context):
        scores = []
        for a in range(self.n):
            try:
                # Compute posterior mean: mu = A^-1 b
                mu = np.linalg.solve(self.A[a], self.b[a])
                
                # Sample theta from N(mu, A^-1)
                # We need to sample from multivariate normal with covariance A^-1.
                # Let A = L L^T (Cholesky). Then A^-1 = L^-T L^-.
                # theta = mu + A^-0.5 * y, where y ~ N(0, I)
                # A^-0.5 = L^-T.
                # So theta = mu + L^-T y.
                
                # Add small jitter for numerical stability if needed, though A should be PD
                # A_reg = self.A[a] + 1e-6 * np.eye(self.d)
                
                L = np.linalg.cholesky(self.A[a])
                y = np.random.randn(self.d)
                
                # Solve L^T w = y  =>  w = (L^T)^-1 y
                w = np.linalg.solve(L.T, y)
                
                theta = mu + w
            except np.linalg.LinAlgError:
                # Fallback: if Cholesky fails, use random or mean only
                # This is rare if A is initialized with lambda*I and updated with xx^T
                theta = np.random.randn(self.d)
            
            scores.append(theta @ context)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Compute prediction and residual
        try:
            mu = np.linalg.solve(self.A[arm], self.b[arm])
            pred = mu @ context
        except np.linalg.LinAlgError:
            pred = 0.0
            mu = np.zeros(self.d)

        residual = reward - pred
        
        # Compute prediction variance
        try:
            # var = x^T A^-1 x
            v = np.linalg.solve(self.A[arm], context)
            var = context @ v
        except np.linalg.LinAlgError:
            var = 1.0

        if var < 1e-6:
            var = 1e-6

        # Standardized innovation
        z = residual / np.sqrt(var)
        z_sq = z * z
        
        # Adaptive forgetting factor
        # Larger z^2 -> smaller gamma -> more forgetting
        gamma = np.exp(-self.beta_forget * z_sq)
        
        # Clamp gamma
        if gamma < self.gamma_min:
            gamma = self.gamma_min
        if gamma > self.gamma_max:
            gamma = self.gamma_max

        # Update precision matrix and vector
        # A_new = gamma * A_old + x x^T
        # b_new = gamma * b_old + r x
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
