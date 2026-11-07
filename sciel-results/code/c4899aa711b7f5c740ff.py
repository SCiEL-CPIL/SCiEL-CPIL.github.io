import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = 280.0  # Safety margin from 300s limit

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_pop = 4 + int(3 * np.log(n))  # Standard population size
        if lambda_pop > budget:
            lambda_pop = max(2, budget) # Ensure at least 1 generation if budget is small
        
        # Weights
        w = np.log(lambda_pop / 2) - np.log(np.arange(1, lambda_pop + 1))
        w = w / np.sum(w)
        mu = int(lambda_pop / 2)
        w = w[:mu]
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2)**2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        # Precompute Cholesky of C (initially I)
        B = np.eye(n)
        D = np.ones(n)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Count evaluations
        evals_used = 0
        best_x = x_mean
        best_f = float('inf')
        
        start_time = time.time()
        
        # Main loop
        while True:
            # Check time limit
            if time.time() > self.deadline:
                break
            
            # Check budget for next generation
            if evals_used + lambda_pop > budget:
                break
            
            # Generate population
            z = np.random.randn(lambda_pop, n)
            
            # Transform to search space
            # y = x_mean + sigma * B @ (D * z)
            z_scaled = z * D
            y = x_mean + sigma * (z_scaled @ B.T)
            
            # Clip to bounds
            y = np.clip(y, -5, 5)
            
            # Evaluate
            f_vals = np.array([func(xi) for xi in y])
            evals_used += lambda_pop
            
            # Update best
            idx_best = np.argmin(f_vals)
            if f_vals[idx_best] < best_f:
                best_f = f_vals[idx_best]
                best_x = y[idx_best].copy()
                
            # Sort by fitness
            order = np.argsort(f_vals)
            y = y[order]
            f_vals = f_vals[order]
            
            # Update mean
            x_new = np.sum(w[:, None] * y[:mu], axis=0)
            
            # Update step size sigma
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mueff) * B.T @ (x_new - x_mean) / (sigma * D)
            delta = (x_new - x_mean) / sigma
            p_sigma_update = B.T @ delta / D
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * p_sigma_update
            
            # Hessian norm check
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)) else 0
            
            sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            
            # Update covariance matrix C
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mueff) * (x_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * delta
            
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i (y_i - x_new)(y_i - x_new)^T
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Term 3: c_mu * sum w_i (y_i - x_new)(y_i - x_new)^T
            dy = y[:mu] - x_new
            weighted_dy = dy * w[:, None]
            C += c_mu * (weighted_dy.T @ dy)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update B and D from C
            try:
                B, D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(n)
                try:
                    B, D = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    break
            
            # Update mean
            x_mean = x_new

        return best_x, best_f
