"""Thompson Sampling with exponentially-weighted posterior for non-stationary linear bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit using Thompson Sampling.
    
    Strategy:
    1. Maintain an exponentially-weighted (sliding-window) posterior for each arm.
       This naturally adapts to non-stationarity.
    2. Use Thompson Sampling: sample theta from the posterior, pick argmax.
    3. Adaptive regularization to prevent numerical issues and encourage exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.99  # forgetting factor for non-stationarity
        # Exponentially weighted sufficient statistics
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive regularization (noise floor)
        self.alpha = 1.0
        self.t = 0
        self._start_time = time.time()
        self._deadline = 140.0  # safety margin before 150s kill

    def _check_deadline(self):
        """Return True if we should stop and return a safe default."""
        return (time.time() - self._start_time) > self._deadline

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        if self._check_deadline():
            # Fallback: return arm with highest mean estimate
            return self._greedy_select(context)

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Compute posterior mean and covariance
            # Posterior: N(theta | mu, Sigma) where
            #   mu = A^{-1} b
            #   Sigma = A^{-1} / alpha  (with adaptive regularization)
            
            # Use Cholesky for stability
            try:
                L = np.linalg.cholesky(self.A[a])
                # Solve L L^T x = b for x = A^{-1} b
                # L^T y = b => y = L^{-T} b
                y = np.linalg.solve(L.T, self.b[a])
                mu = np.linalg.solve(L, y)
                
                # Sample from N(mu, A^{-1} / alpha)
                # A^{-1} = L^{-1} L^{-T}
                # So A^{-1}/alpha = (1/sqrt(alpha)) L^{-1} L^{-T} (1/sqrt(alpha))
                # Sample: mu + (1/sqrt(alpha)) * L^{-1} * z, where z ~ N(0, I)
                z = np.random.standard_normal(self.d)
                # Solve L v = z => v = L^{-1} z
                v = np.linalg.solve(L, z)
                theta_sample = mu + (1.0 / np.sqrt(self.alpha)) * v
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                try:
                    inv_A = np.linalg.inv(self.A[a])
                    mu = inv_A @ self.b[a]
                    z = np.random.standard_normal(self.d)
                    theta_sample = mu + (1.0 / np.sqrt(self.alpha)) * (inv_A @ z)
                except:
                    theta_sample = self.b[a] / (self.alpha + 1e-6)

            scores[a] = theta_sample @ context

        return int(np.argmax(scores))

    def _greedy_select(self, context):
        """Safe fallback: pick arm with highest posterior mean."""
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        for a in range(self.n):
            try:
                mu = np.linalg.solve(self.A[a], self.b[a])
                scores[a] = mu @ context
            except:
                scores[a] = 0.0
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Apply forgetting: decay old information
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Adaptively adjust regularization:
        # Increase alpha when we're early (less data) or when weights change rapidly
        # Use a simple heuristic: alpha decreases as we collect more effective samples
        # Effective sample size ~ 1 / (1 - gamma) for steady state
        # Start high for exploration, decay toward a floor
        self.alpha = max(0.5, 1.0 / (1.0 + 0.1 * self.t))
# EVOLVE-BLOCK-END
