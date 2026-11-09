import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val
        
        def cma_es_run(x_init, sigma_init, max_evals, rng):
            nonlocal calls_made, best_x, best_f
            
            # CMA-ES parameters
            lam = 10  # Larger population for better diversity
            mu = max(1, lam // 2)  # Number of selected solutions
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Adaptation parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            delta_hc = 1.0  # Simplified
            
            # State
            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            evals_in_run = 0
            best_f_run = float('inf')
            no_improve_count = 0
            
            while evals_in_run < max_evals:
                if time.time() > deadline:
                    break
                
                # Generate population
                xs = np.zeros((lam, dim))
                fs = np.zeros(lam)
                
                # Compute sqrt(C) via eigen decomposition
                w, V = np.linalg.eigh(C)
                w = np.clip(w, 1e-12, None)
                sqrt_w = np.sqrt(w)
                sqrt_C = (V * sqrt_w) @ V.T  # V @ diag(sqrt_w) @ V.T
                
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_new = x_mean + sigma * (sqrt_C @ z)
                    x_new = np.clip(x_new, -5, 5)
                    
                    f_new = evaluate(x_new)
                    if f_new is None:
                        return
                    xs[i] = x_new
                    fs[i] = f_new
                    evals_in_run += 1
                    if f_new < best_f_run:
                        best_f_run = f_new
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                
                if evals_in_run == 0:
                    break
                
                # Sort by fitness
                sort_idx = np.argsort(fs)
                xs_sorted = xs[sort_idx]
                fs_sorted = fs[sort_idx]
                
                # Check for stagnation
                if no_improve_count > lam * 3:
                    break
                
                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # Update p_sigma
                inv_sqrt_w = 1.0 / sqrt_w
                inv_sqrt_C = (V * inv_sqrt_w) @ V.T
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sigma = 1.0 if norm_p_sigma < (1 + 0.3 * min(1, 1.0 / mu_eff)) * (np.sqrt(mu_eff) + 1) * np.sqrt(c_sigma * (2 - c_sigma)) else 0.0
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C @ (x_mean - x_old) / sigma
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update p_c
                norm_p_c = np.linalg.norm(p_c)
                h_c = 1.0 if norm_p_c < (1 + 0.3 * min(1, 1.0 / mu_eff)) * (np.sqrt(mu_eff) + 1) * np.sqrt(c_c * (2 - c_c)) else 0.0
                p_c = (1 - c_c) * p_c + c_c * h_c * (x_mean - x_old) / sigma
                
                # Update C
                rank_one = np.outer(p_c, p_c)
                # rank_mu: weighted sum of outer products of top mu solutions
                diffs = xs_sorted[:mu] - x_old
                rank_mu = np.einsum('i,ij,ik->jk', weights, diffs, diffs)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - delta_hc * h_c) * rank_mu) + c_mu * rank_mu
                
                # Symmetrize and ensure positive definiteness
                C = (C + C.T) / 2
                C += 1e-10 * np.eye(dim)
                
                # Check convergence
                if sigma < 1e-6:
                    break
            
            return
        
        # Multiple restart strategy for multimodal Rastrigin
        rng = np.random.default_rng(42)
        
        # First run: use ~60% of budget
        first_budget = int(budget * 0.6)
        cma_es_run(np.zeros(dim), 2.5, first_budget, rng)
        
        # If we have budget left, do restarts from best point with smaller sigma
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Restart from best point with perturbation
            perturbation = 0.5 * np.ones(dim)
            x_restart = np.clip(best_x + rng.standard_normal(dim) * perturbation, -5, 5)
            
            # Use remaining budget for this run
            remaining = budget - calls_made
            cma_es_run(x_restart, 0.8, remaining, rng)
            
            if calls_made >= budget:
                break
        
        return best_x, best_f
