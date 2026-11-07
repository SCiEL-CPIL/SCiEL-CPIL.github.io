"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 290s to be safe within 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        
        # Initialize mean at origin (minimizer is in [-4,4]^5, origin is a good prior)
        x_mean = np.zeros(dim)
        f_mean = func(x_mean)
        best_x = x_mean.copy()
        best_f = f_mean
        
        # Initial sigma: cover a good portion of the search space
        sigma = 2.0
        
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper for matrix square root inverse
        def mat_sqrt_inv(M):
            w, V = np.linalg.eigh(M)
            w = np.clip(w, 1e-20, None)
            return (V @ np.diag(1.0 / np.sqrt(w)) @ V.T)
        
        def mat_sqrt(M):
            w, V = np.linalg.eigh(M)
            w = np.clip(w, 1e-20, None)
            return (V @ np.diag(np.sqrt(w)) @ V.T)
        
        for i in range(self.budget):
            if time.time() > deadline:
                break
                
            # Generate population
            pop_x = []
            pop_f = []
            
            for j in range(lambda_pop):
                # Generate z
                z = rng.standard_normal(dim)
                
                # Transform z using C_inv_sqrt
                z_c = C_inv_sqrt @ z
                
                # Step
                x_new = np.clip(x_mean + sigma * z_c, -5, 5)
                
                # Evaluate
                f_new = func(x_new)
                
                pop_x.append(x_new)
                pop_f.append(f_new)
                
                # Update best
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            
            # Sort by fitness
            indices = np.argsort(pop_f)
            sorted_x = [pop_x[k] for k in indices]
            sorted_f = [pop_f[k] for k in indices]
            
            # Compute weighted mean of mu best
            x_new_mean = np.zeros(dim)
            for j in range(mu):
                x_new_mean += weights[j] * sorted_x[j]
            
            # Update paths
            # Need C_inv_sqrt @ (x_new_mean - x_mean) / sigma
            diff = (x_new_mean - x_mean) / sigma
            p_sigma += c_sigma * (C_inv_sqrt @ diff - (1 - c_sigma) * C_inv_sqrt @ diff) # Simplified, standard CMA uses h_sigma check
            
            # Proper CMA-ES update
            # h_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (i + 1)))) / chi_n - (1 - c_sigma)
            h_sigma = 1 if h_sigma < 1.4 else 0 # Approximation
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * c_sigma * (C_inv_sqrt @ diff)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            
            # Update covariance
            delta = (1 + (1 - h_sigma) * c_mu * mu_eff) * C
            delta += c_mu * (np.outer(p_c, p_c) - C)
            
            # Rank-mu update
            for j in range(mu):
                y_j = (sorted_x[j] - x_mean) / sigma
                delta += c_mu * weights[j] * np.outer(y_j, y_j)
            
            C = delta
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-10):
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update mean
            x_mean = x_new_mean
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C_inv_sqrt
            C_inv_sqrt = mat_sqrt_inv(C)
            
            # Update f_mean if we want to track it, but we track best separately
            # f_mean = sorted_f[0] # Not strictly needed for best tracking
            
        return best_x, best_f
# EVOLVE-BLOCK-END
