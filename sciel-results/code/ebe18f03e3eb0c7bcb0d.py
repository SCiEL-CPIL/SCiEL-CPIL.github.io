"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with covariance adaptation and Nelder-Mead polish for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        box_lo, box_hi = -5.0, 5.0

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            x = np.clip(x, box_lo, box_hi)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diverse random initialization
        n_init = min(30, budget // 8)
        init_points = []
        for i in range(n_init):
            x = rng.uniform(box_lo, box_hi, dim)
            fx = evaluate(x)
            init_points.append((fx, x))
        
        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Sort initial points and pick top few for multi-start CMA-ES
        init_points.sort(key=lambda t: t[0])
        
        # Phase 2: CMA-ES with covariance matrix adaptation
        # We'll run CMA-ES from the best initial point, then restart from other good points if needed
        
        def run_cma_es(start_x, start_f, sigma_init, remaining_budget):
            nonlocal calls_used, best_f, best_x
            if remaining_budget <= 0:
                return start_x, start_f
            
            dim_local = dim
            lambda_ = 4 + int(3 * np.log(dim_local))  # ~8 for dim=5
            mu = lambda_ // 2
            
            # CMA-ES parameters
            cc = (4 + dim_local / (dim_local + 4)) / ((dim_local + 2) + 4 + dim_local / (dim_local + 4))
            cs = (dim_local + 1.7) / (dim_local + cs + 3.3)
            c1 = 2 / ((dim_local + 1.3)**2 + mu)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu) / ((dim_local + 2)**2 + mu))
            mu_eff = 1 / (1 / mu)  # effective population size
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            
            # State
            x_mean = start_x.copy()
            sigma = sigma_init
            C = np.eye(dim_local)
            pc = np.zeros(dim_local)
            ps = np.zeros(dim_local)
            
            # Eigen decomposition cache
            eigvals, eigvecs = np.linalg.eigh(C)
            D = 1.0 / np.sqrt(eigvals)
            B = eigvecs
            
            calls_local = 0
            stagnation_counter = 0
            last_best_f = start_f
            
            while calls_local < remaining_budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                for j in range(lambda_):
                    if calls_local >= remaining_budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Sample from N(0, C)
                    z = rng.standard_normal(dim_local)
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, box_lo, box_hi)
                    
                    fx_new = func(x_new)
                    calls_local += 1
                    offspring.append((fx_new, x_new))
                
                if not offspring:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()
                
                # Check for progress
                if offspring[0][0] < last_best_f - 1e-10:
                    stagnation_counter = 0
                    last_best_f = offspring[0][0]
                else:
                    stagnation_counter += 1
                
                # Weighted recombination
                y_w = np.zeros(dim_local)
                for i in range(mu):
                    y_w += w[i] * (offspring[i][1] - x_mean) / sigma
                
                # Update path accumulators
                pc = (1 - cc) * pc + cc * y_w
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D * y_w))
                
                # Update mean
                x_mean = x_mean + sigma * y_w
                
                # Update covariance
                delta_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - 1) * cs * c1
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim_local, dim_local))
                for i in range(mu):
                    diff = (offspring[i][1] - x_mean + sigma * y_w) / sigma
                    rank_mu += w[i] * np.outer(diff, diff)
                
                C = (1 + c1 * delta_sig - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                
                # Ensure positive definite
                eigvals_new, eigvecs_new = np.linalg.eigh(C)
                eigvals_new = np.maximum(eigvals_new, 1e-20)
                C = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
                eigvals, eigvecs = eigvals_new, eigvecs_new
                D = 1.0 / np.sqrt(eigvals)
                B = eigvecs
                
                # Step size adaptation
                sigma *= np.exp(cs / sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Restart condition
                if stagnation_counter > 10 and calls_local < remaining_budget * 0.8:
                    break
            
            return x_mean, best_f if best_f < start_f else start_f

        # Run CMA-ES from best initial point with remaining budget
        cma_budget = budget - calls_used - 10  # reserve some for polish
        cma_budget = max(cma_budget, 50)
        
        _, _ = run_cma_es(best_x.copy(), best_f, 1.5, cma_budget)
        
        # Phase 3: Nelder-Mead polish from best point found
        if calls_used < budget and time.time() < deadline:
            try:
                res = minimize(func, best_x, method='Nelder-Mead', 
                              options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10},
                              bounds=[(box_lo, box_hi)] * dim)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, box_lo, box_hi).copy()
            except Exception:
                pass
        
        return (best_x, best_f)
# EVOLVE-BLOCK-END
