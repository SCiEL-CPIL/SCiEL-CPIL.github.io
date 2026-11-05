import numpy as np
import time

class Policy:
    """
    Adaptive Discounted Thompson Sampling for Non-Stationary Linear Bandits.
    
    Uses a Gaussian posterior over arm weights with a discount factor to handle non-stationarity.
    The discount factor is adapted based on recent prediction errors to balance exploration and exploitation.
    """
    
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Initialize sufficient statistics for each arm
        # A[a] is the precision matrix (inverse covariance) scaled by prior precision
        # b[a] is the sufficient statistic for the mean
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive discount factor
        self.beta = 0.99
        self.beta_min = 0.95
        self.beta_max = 0.999
        self.beta_lr = 0.001
        
        # Error tracking for adaptive beta
        self.recent_errors = []
        self.error_window = 20
        self.error_threshold = 1.0 # Target prediction error
        
        # Time tracking for deadline safety
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        
        # Pre-compute identity for quick reset if needed
        self.eye = np.eye(dim)

    def select(self, context):
        # Check deadline safety margin
        if time.time() > self.deadline:
            # Fallback: exploit best known mean
            scores = [self.A[a] and (np.linalg.solve(self.A[a], self.b[a]) @ context) or 0.0 for a in range(self.n_arms)]
            return int(np.argmax(scores))

        context = np.asarray(context)
        best_arm = -1
        best_score = -np.inf
        
        for a in range(self.n_arms):
            # Compute posterior mean: mu = A^{-1} b
            try:
                mu = np.linalg.solve(self.A[a], self.b[a])
                # Compute posterior covariance: Sigma = A^{-1}
                # Use Cholesky for sampling stability
                L = np.linalg.cholesky(self.A[a])
                # Sample z ~ N(0, I)
                z = np.random.randn(self.dim)
                # Solve L * v = z => v = L^{-1} z
                v = np.linalg.solve(L, z)
                # Sigma = (L L^T)^{-1} = L^{-T} L^{-1}
                # Sample theta = mu + Sigma^{1/2} z = mu + L^{-T} v
                # Note: A = L L^T is not standard for precision. 
                # Standard Gaussian: p(theta) ~ N(mu, Sigma). 
                # If we maintain A as precision (inverse covariance), then Sigma = A^{-1}.
                # Cholesky of A: A = L L^T.
                # We want sample theta ~ N(mu, A^{-1}).
                # theta = mu + A^{-1/2} z.
                # A^{-1/2} = L^{-T}.
                # So theta = mu + L^{-T} z.
                # We computed v = L^{-1} z. We need L^{-T} z.
                # Let's just use np.linalg.inv for simplicity if dim is small, or proper Cholesky solve.
                # Let's stick to simple inverse for dim <= 20 as it's robust enough and fast enough for 5 arms.
                inv_A = np.linalg.inv(self.A[a])
                sigma = inv_A
                theta_sample = mu + np.random.multivariate_normal(np.zeros(self.dim), sigma)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                theta_sample = self.b[a] / (np.sum(self.b[a]**2) + 1e-6)
                
            score = theta_sample @ context
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # Update sufficient statistics with discount
        self.A[arm] = self.beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.beta * self.b[arm] + reward * context
        
        # Adaptive Beta Update
        # Compute prediction error for the chosen arm
        try:
            mu = np.linalg.solve(self.A[arm], self.b[arm])
            pred = mu @ context
        except np.linalg.LinAlgError:
            pred = 0.0
            
        error = (reward - pred) ** 2
        self.recent_errors.append(error)
        
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)
            
        if len(self.recent_errors) == self.error_window:
            avg_error = np.mean(self.recent_errors)
            # If error is high, decrease beta (forget faster)
            # If error is low, increase beta (remember longer)
            adjustment = self.beta_lr * (self.error_threshold - avg_error)
            self.beta = np.clip(self.beta + adjustment, self.beta_min, self.beta_max)
