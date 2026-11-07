"""Optimization using CMA-ES for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        # CMA-ES Parameters
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        # Number of recombination weights
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        
        # Effective population size
        mueff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0 # Initial step size, box is [-5,5]
        C = np.eye(dim)
        B = np.eye(dim) # Eigenvectors
        D = np.ones(dim) # Eigenvalues (sqrt of eigenvalues of C)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute C^0.5
        C_sqrt = B * D
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0
        
        # Initial evaluation
        # We start with the mean, but CMA-ES usually samples around it.
        # To be efficient, let's just run the loop.
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate samples
            # We need to generate lambda_ samples.
            # If remaining budget < lambda_, we might need to adjust or stop.
            if budget - calls < lambda_:
                # If we don't have enough for a full population, 
                # we can do a smaller population or stop.
                # For simplicity and robustness, if we can't do a full pop, 
                # we might do a (1+1) step or just break.
                # Given lambda=10 and budget 500, we have 50 generations.
                # If we are at call 495, we have 5 calls left. 
                # Let's just stop if we can't do a full population to maintain CMA-ES integrity.
                # Or we can reduce lambda_.
                # Let's just break if we can't complete a generation.
                break
            
            Z = np.random.randn(lambda_, dim)
            X = np.tile(x_mean, (lambda_, 1)) + sigma * (Z @ C_sqrt.T)
            
            # Clip to box [-5, 5]
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lambda_
            
            # Track best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            # Sort indices
            idx_sorted = np.argsort(F)
            X_sorted = X[idx_sorted]
            F_sorted = F[idx_sorted]
            
            # Update mean
            x_new = np.dot(w, X_sorted[:mu])
            
            # Update sigma
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt.T @ (x_new - x_mean) / sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) - np.sqrt(dim) * (1 - 1/np.sqrt(2))))
            
            # Update C
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_mean) / sigma
            
            # Rank-1 update
            C = (1 + c_1 * (c_c * 2 + 2 - c_c) - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # y = (X_sorted[:mu] - x_mean) / sigma
            Y = (X_sorted[:mu] - x_mean) / sigma
            # C = (1 + c_mu * (c_c * 2 + 2 - c_c) - c_mu) * C + c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', Y, Y), axis=0)
            # More efficient:
            # C += c_mu * (np.dot(w[:, None] * Y.T, Y))
            # Wait, the formula is sum_i w_i * y_i * y_i^T
            C += c_mu * np.sum(w[:, np.newaxis, np.newaxis] * Y[:, np.newaxis, :] * Y[:, :, np.newaxis], axis=0)
            
            # Eigen decomposition
            # To avoid numerical issues, we do this periodically or every step.
            # For dim=5, it's cheap.
            C, B, D = self._eigen_decompose(C)
            C_sqrt = B * D
            
            # Move mean
            x_mean = x_new
            
            # Prevent sigma from becoming too small
            if sigma < 1e-8:
                break

        return best_x, best_f

    def _eigen_decompose(self, C):
        """Stable eigen decomposition."""
        # Symmetrize
        C = 0.5 * (C + C.T)
        D, B = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        D = np.maximum(D, 1e-12)
        D = np.sqrt(D)
        return C, B, D
# EVOLVE-BLOCK-END
