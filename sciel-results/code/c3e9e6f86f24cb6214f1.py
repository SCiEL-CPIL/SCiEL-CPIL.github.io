"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting and Cholesky-based variance for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_forgetting = 0.995
        self.current_forgetting = self.base_forgetting
        
        # EWMA of squared residuals for non-stationarity detection
        self.residual_sq_ewma = [0.0 for _ in range(n_arms)]
        self.residual_ewma_alpha = 0.05
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 120.0  # Safety margin before 150s kill

    def _get_forgetting(self, arm, context, reward):
        """Adapt forgetting rate based on observed non-stationarity."""
        if self.t > 0:
            # Compute residual
            inv_A = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(self.d))
            theta = inv_A @ self.b[arm]
            pred = theta @ context
            residual = reward - pred
            residual_sq = residual ** 2
            
            # Update EWMA of squared residuals
            self.residual_sq_ewma[arm] = (1 - self.residual_ewma_alpha) * self.residual_sq_ewma[arm] + self.residual_ewma_alpha * residual_sq
            
            # If residual is large, increase forgetting (lower alpha)
            # Base forgetting is 0.995, max forgetting is 0.9 (more aggressive)
            # Scale based on residual magnitude
            if self.residual_sq_ewma[arm] > 0.1:
                # More aggressive forgetting when non-stationarity detected
                self.current_forgetting = max(0.9, self.base_forgetting - 0.1 * np.sqrt(self.residual_sq_ewma[arm]))
            else:
                # Gradually return to base forgetting
                self.current_forgetting += 0.001 * (self.base_forgetting - self.current_forgetting)
        
        return self.current_forgetting

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return the arm with highest mean reward so far
            means = [np.linalg.inv(self.A[a] + 1e-6 * np.eye(self.d)) @ self.b[a] @ context for a in range(self.n)]
            return int(np.argmax(means))
        
        scores = []
        for a in range(self.n):
            # Use Cholesky for stable variance computation
            try:
                L = np.linalg.cholesky(self.A[a] + 1e-6 * np.eye(self.d))
                inv_A = np.linalg.inv(self.A[a] + 1e-6 * np.eye(self.d))
                theta = inv_A @ self.b[a]
                mean = theta @ context
                # Variance: context^T A^{-1} context = ||A^{-1/2} context||^2
                v = np.linalg.solve(L, context)
                variance = np.dot(v, v)
                scores.append(mean + 0.5 * np.sqrt(variance))
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                inv_A = np.linalg.inv(self.A[a] + 1e-6 * np.eye(self.d))
                theta = inv_A @ self.b[a]
                mean = theta @ context
                variance = context @ inv_A @ context
                scores.append(mean + 0.5 * np.sqrt(variance))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Get adaptive forgetting rate
        alpha = self._get_forgetting(arm, context, reward)
        
        # Update with forgetting
        self.A[arm] = alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = alpha * self.b[arm] + reward * context
        
        self.t += 1
# EVOLVE-BLOCK-END
