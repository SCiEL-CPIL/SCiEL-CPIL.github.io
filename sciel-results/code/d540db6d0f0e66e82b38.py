# EVOLVE-BLOCK-START
"""
CMA-ES Optimizer for BBOB Sphere (f1).

Strategy:
1. Implement CMA-ES (Covariance Matrix Adaptation Evolution Strategy), a state-of-the-art
   derivative-free evolutionary strategy that adapts both the step size and the shape of
   the search distribution (via a covariance matrix) to the local landscape.
2. CMA-ES is particularly effective on smooth, convex, and symmetric functions like Sphere,
   as it can quickly align the search direction with the principal axes of the function's
   level sets.
3. Use a standard CMA-ES implementation with adaptive step size and covariance matrix
   updates. This avoids the local minima traps and fixed restart triggers of the previous
   Nelder-Mead + gradient polish approach, providing more robust global convergence.
4. Budget management: Use the full budget for CMA-ES iterations, with a safety margin
   for the 300s wall-clock limit.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # CMA-ES parameters
        # Population size: 4 + 3 * log(dim) for dim=5 -> 4 + 3*1.6 = 8.8 -> 9
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial mean
        x_mean = np.zeros(dim)
        
        # Initial step size
        # For Sphere on [-5, 5], a reasonable initial step size is ~1.0
        sigma = 1.0
        
        # Initial covariance matrix (identity)
        C = np.eye(dim)
        
        # Eigen decomposition for sampling
        # D is diagonal matrix of sqrt(eigenvalues)
        # B is matrix of eigenvectors
        # We maintain C = B @ D^2 @ B.T
        # And sample y ~ N(0, I), then z = B @ (D * y)
        
        # Initialize B, D from C
        eigvals, eigvecs = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(eigvals, 1e-12))
        B = eigvecs
        
        # Parameters for adaptation
        # Learning rates
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 + 1) / (dim + 2) - 1))
        
        # Weight vector for selection
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Step size control parameters
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        h_sigma = 0.0
        
        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Main loop
        while calls < budget and time.time() < deadline:
            # Generate population
            pop_x = np.zeros((lambda_, dim))
            pop_f = np.zeros(lambda_)
            
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                # Sample y ~ N(0, I)
                y = np.random.normal(0, 1, dim)
                # Transform to z
                z = B @ (D * y)
                # New point
                x_i = x_mean + sigma * z
                # Clip to box
                x_i = np.clip(x_i, -5, 5)
                
                f_i = func(x_i)
                calls += 1
                pop_x[i] = x_i
                pop_f[i] = f_i
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x = pop_x[order]
            pop_f = pop_f[order]
            
            # Selection: take best mu individuals
            x_w = pop_x[:mu]
            
            # New mean
            x_mean_new = np.sum(weights[:, np.newaxis] * x_w, axis=0)
            
            # Update step size
            y_old = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B.T @ y_old
            
            # Hessianian norm check for step size update
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + c_c * B.T @ y_old
            
            # Rank-one update
            C = (1 + c_1 * (mu_eff - 1) - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            C += c_mu * np.sum(weights[i] * np.outer(x_w[i] - x_mean, x_w[i] - x_mean) for i in range(mu))
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Eigen decomposition for next iteration
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
            except:
                # Fallback if eigen decomposition fails
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
            
            x_mean = x_mean_new
            
            # Check convergence
            if np.all(np.abs(D) < 1e-10):
                break
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
