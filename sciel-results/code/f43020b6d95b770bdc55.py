import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 280 seconds
        deadline = start_time + 280
        
        n = self.dim
        # CMA-ES Parameters
        # Population size
        lambda_ = 10
        # Number of points used for mean update
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (n + 4)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initialization
        rng = np.random.default_rng(0)
        x_mean = np.zeros(n)
        sigma = 2.0 # Initial step size
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        eval_count = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        eval_count += 1
        
        # Main Loop
        while eval_count < self.budget:
            if time.time() > deadline:
                break
                
            # Sample points
            z = rng.standard_normal((lambda_, n))
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Reset if covariance matrix becomes invalid
                C = np.eye(n)
                L = np.eye(n)
                sigma = 2.0
                x_mean = np.zeros(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                continue
                
            # Generate population
            x_pop = x_mean[None, :] + sigma * (z @ L.T)
            
            # Clip to bounds
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate fitness
            f_pop = np.array([func(x) for x in x_pop])
            eval_count += lambda_
            
            # Update best
            idx_min = np.argmin(f_pop)
            if f_pop[idx_min] < best_f:
                best_f = f_pop[idx_min]
                best_x = x_pop[idx_min].copy()
                
            # Sort population by fitness
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update evolution paths
            step = (x_mean_new - x_mean) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.solve(L, step)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            v = (x_sorted[:mu] - x_mean) / sigma
            outer = np.sum(weights[:, None, None] * v[:, None, :] * v[:, :, None], axis=0)
            
            # Heuristic check for path norms to avoid divergence
            # Simple version: always update
            C = (1 - c_1 - c_mu) * C + c_1 * (pc[:, None] * pc[None, :] + (1 - c_c * (2 - c_c)) * np.eye(n)) + c_mu * outer
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definiteness
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-10):
                C = C + 1e-10 * np.eye(n)
                
            x_mean = x_mean_new

        return best_x, best_f
