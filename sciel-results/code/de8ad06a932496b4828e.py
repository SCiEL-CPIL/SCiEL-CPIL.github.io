"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        # CMA-ES parameters
        dim = self.dim
        budget = self.budget
        
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        
        # Initial step size
        x0 = np.zeros(dim)
        sigma = 1.0
        
        # Initial covariance matrix
        C = np.eye(dim)
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial evaluation
        x = x0.copy()
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Precompute Cholesky factor
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(dim)
            
        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                # Reflect if out of bounds
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate
            fs = []
            for y in ys:
                fy = func(y)
                fs.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            # Sort
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update mean
            x_new = sum(w * y for w, y in zip(weights, ys[:mu]))
            
            # Update step size
            y_sigma = (C_sqrt.T @ p_sigma + x_new - x) / sigma
            sigma *= np.exp(c_sigma * (np.linalg.norm(y_sigma) - np.sqrt(dim) * (1 - 1 / (2 * dim))) / dim)
            
            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt.T @ (x_new - x)) / sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**((calls + 1) * c_sigma)) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)) else 0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_new - x) / sigma
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w * np.outer(y - x, y - x) for w, y in zip(weights, ys[:mu]))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
            
            x = x_new
            fx = best_f # Approximation, we use best_f for tracking
            
        return best_x, best_f

# EVOLVE-BLOCK-END
