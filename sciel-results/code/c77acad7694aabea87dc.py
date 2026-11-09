"""Rewritten program for BBOB f7 (Step ellipsoidal) using Multi-start CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # Parameters for CMA-ES
        # Population size lambda
        lambda_ = 4 + int(3 * np.log(dim))  # For dim=5, lambda=14
        # Number of restarts
        # We have 500 budget. Let's do 5 restarts of 100 evals each.
        num_restarts = 5
        budget_per_restart = budget // num_restarts
        
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize CMA-ES state
            # Random center
            x0 = rng.uniform(-5, 5, dim)
            sigma = 2.0
            C = np.eye(dim)
            # Step size for C (inverse sqrt of eigenvalues, initialized to 1)
            # Standard CMA-ES parameters
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            ps = 0.0
            
            # Run CMA-ES for budget_per_restart
            for _ in range(budget_per_restart):
                if time.time() > deadline:
                    break
                
                # Sample lambda points
                Y = rng.multivariate_normal(np.zeros(dim), C, size=lambda_)
                X = x0 + sigma * Y
                # Clip to box
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([func(x) for x in X])
                
                # Update best
                idx_min = np.argmin(F)
                if F[idx_min] < best_f:
                    best_f = F[idx_min]
                    best_x = X[idx_min].copy()
                    
                # Sort
                idx_sort = np.argsort(F)
                X_sorted = X[idx_sort]
                F_sorted = F[idx_sort]
                
                # Mean
                y_w = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
                y_w_sigma = y_w / sigma
                
                # Cumulation
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_N * y_w_sigma
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * chi_N else 0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_w_sigma
                
                # Covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                    + c_1 / (1 - (1 - c_c)**2) * np.outer(p_c, p_c) \
                    + c_mu / (1 - (1 - c_c)**2) * np.sum(weights[:mu, None] * np.outer(X_sorted[:mu] - x0, X_sorted[:mu] - x0) / sigma**2, axis=0)
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                # Eigen decomposition for sampling
                try:
                    D, V = np.linalg.eigh(C)
                    D = np.maximum(D, 1e-12) # Numerical stability
                    # Update sigma
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
                    # Cap sigma
                    sigma = np.clip(sigma, 1e-8, 10.0)
                    x0 = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
                except np.linalg.LinAlgError:
                    # Fallback if eigen fails
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
