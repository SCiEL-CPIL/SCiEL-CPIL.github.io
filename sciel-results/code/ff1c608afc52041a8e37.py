"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def run_cmaes(x0, sigma0, budget_for_run):
            nonlocal best_x, best_f
            
            # Parameters for dim=5
            lambda_ = 8
            mu = 4
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = (w.sum() ** 2) / np.sum(w ** 2)
            
            # Adaptation parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 - mu_eff / mu_eff) / (dim + 4) # Using 0.33 for dim=5
            c_c = 0.33
            c_mu = (mu_eff - 2) / (dim + mu_eff + 1)
            c_cov = (2 + 2 * c_mu) / (dim + 1.3 + c_mu)
            
            # Initialization
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            calls_used = 0
            stagnation_counter = 0
            last_best_f = best_f
            
            while calls_used < budget_for_run:
                if time.time() > deadline:
                    break
                
                # Sample lambda points
                try:
                    L = np.linalg.cholesky(C)
                except:
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                offspring = []
                for j in range(lambda_):
                    if calls_used >= budget_for_run:
                        break
                    if time.time() > deadline:
                        break
                    
                    y = L @ rng.standard_normal(dim)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    
                    fx = func(x_new)
                    calls_used += 1
                    offspring.append((fx, x_new, y))
                
                if not offspring:
                    break
                    
                # Sort
                offspring.sort(key=lambda t: t[0])
                
                # Update global best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                if stagnation_counter > 30:
                    break
                    
                # Calculate weighted mean of selected solutions
                z_w = np.zeros(dim)
                for i in range(mu):
                    z_w += w[i] * offspring[i][2]
                    
                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = x_mean + c_mu * sigma * z_w
                x_mean = np.clip(x_mean, -5, 5)
                
                # Update step size
                y_sigma = (np.linalg.norm(z_w) / np.sqrt(mu_eff)) * (1 / np.sqrt(np.trace(C)))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(2 * dim) - 1))
                
                # Update C
                # p_c update
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_old) / sigma
                
                # C update
                H = (np.outer(p_c, p_c) / (1 - (1 - c_c) ** (2 * calls_used))) - c_cov / (dim + 1) * np.eye(dim)
                C = (1 + c_cov * (2 - c_cov) * mu_eff - c_cov) * C + c_cov * (
                    np.outer(p_c, p_c) / (1 - (1 - c_c) ** (2 * calls_used)) - c_cov / (dim + 1) * np.eye(dim)
                )
                
                # Ensure C is positive definite
                if not np.all(np.linalg.eigvalsh(C) > 0):
                    C = np.eye(dim)

        # Multi-start strategy
        n_runs = 3
        budget_per_run = total_budget // n_runs
        
        centers = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim)
        ]
        
        for i in range(n_runs):
            if time.time() > deadline:
                break
            if i * budget_per_run >= total_budget:
                break
                
            x0 = centers[i]
            x0 = np.clip(x0, -5, 5)
            
            run_cmaes(x0, 1.0, budget_per_run)
            
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return (best_x, best_f)
# EVOLVE-BLOCK-END
