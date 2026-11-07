# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f7 (Step Ellipsoidal) using a hybrid strategy:
    1. Global exploration with a simplified, robust CMA-ES variant.
    2. Local polishing with L-BFGS-B (better conditioned for smooth regions) 
       or Nelder-Mead if L-BFGS-B fails.
    
    The Step Ellipsoidal function has a flat region around the global minimum 
    and a steep valley. CMA-ES helps navigate the valley, while gradient-based 
    methods (L-BFGS-B) can exploit the smoothness near the minimum for rapid 
    convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 290s to be safe.
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_made += 1

        def time_check():
            return time.time() > deadline or calls_made >= budget

        # --- CMA-ES Implementation ---
        def run_cma_es(initial_mean, sigma_init, max_generations=None):
            nonlocal calls_made, best_x, best_f
            
            if time_check():
                return

            # Standard CMA-ES parameters for dim=5
            lam = 4 + int(3 * np.log(dim))  # 8 for dim=5
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            mean = initial_mean.copy()
            sigma = sigma_init
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)

            gen = 0
            max_gen = max_generations if max_generations else 100
            stagnant_count = 0
            last_best_f = best_f

            while gen < max_gen and not time_check():
                # Decompose C for sampling
                # Use eigh for symmetric C
                val, vec = np.linalg.eigh(C)
                val = np.clip(val, 1e-12, None)
                sqrt_val = np.sqrt(val)
                # inv_sqrt_C @ diff is equivalent to vec @ (inv_sqrt_val * (vec.T @ diff))
                # But for sampling x = mean + sigma * (vec @ (sqrt_val * z))
                
                pop = np.empty((lam, dim))
                fvals = np.empty(lam)
                
                for i in range(lam):
                    if time_check():
                        fvals[i] = np.inf
                        continue
                    z = rng.standard_normal(dim)
                    x_i = mean + sigma * (vec @ (z * sqrt_val))
                    np.clip(x_i, -5, 5, out=x_i)
                    pop[i] = x_i
                    fvals[i] = func(x_i)
                    calls_made += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = x_i.copy()
                        stagnant_count = 0
                    else:
                        stagnant_count += 1 # Per sample stagnation? No, per generation is better.
                
                if time_check():
                    break

                # Sort
                order = np.argsort(fvals)
                sorted_pop = pop[order]
                sorted_fvals = fvals[order]
                
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # Update paths
                diff = (new_mean - mean) / sigma
                # inv_sqrt_C @ diff
                # vec @ ( (1/sqrt_val) * (vec.T @ diff) )
                tmp = vec.T @ diff
                tmp = tmp / sqrt_val
                inv_sqrt_C_diff = vec @ tmp
                
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * inv_sqrt_C_diff
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C_diff
                
                # Update sigma
                norm_ps = np.linalg.norm(ps)
                delta_sigma = (norm_ps / chi_n - 1) * d_sigma
                sigma *= np.exp(c_sigma * delta_sigma)
                sigma = np.clip(sigma, 1e-12, 10.0)
                
                # Update covariance
                y = (sorted_pop[:mu] - mean) / sigma
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(y[i], y[i])
                
                decay = 1 - c_1 - c_mu
                rank_one = np.outer(pc, pc) + (delta_sigma * c_sigma * (2 - c_sigma)) * C
                C = decay * C + c_1 * rank_one + c_mu * rank_mu
                C = (C + C.T) / 2
                
                mean = new_mean
                gen += 1
                
                # Stagnation check: If best_f hasn't improved in last 20 generations
                if best_f < last_best_f - 1e-10:
                    last_best_f = best_f
                    stagnant_count = 0
                else:
                    stagnant_count += 1
                    
                if stagnant_count > 20:
                    break

        # --- Strategy ---
        # 1. Initial CMA-ES run from origin with moderate sigma
        # BBOB f7 minimizer is in [-4, 4]^5. Origin is a reasonable start.
        # Sigma 1.5 covers a good portion of the box.
        run_cma_es(np.zeros(dim), 1.5, max_generations=40)

        # 2. If we have budget left, do a local polish.
        # L-BFGS-B is often faster and more robust for smooth convex-like regions 
        # than Nelder-Mead, especially in low dimensions.
        # We wrap func to count calls and enforce bounds.
        
        if not time_check():
            # Use L-BFGS-B with bounds
            # Note: L-BFGS-B in scipy may not respect bounds perfectly if it steps out, 
            # but it handles them. We also clip in the wrapper just in case? 
            # Actually, scipy's L-BFGS-B handles bounds internally.
            
            # We need to be careful with the number of calls.
            # L-BFGS-B can be expensive if maxiter is high.
            # Let's reserve ~30% of budget for polishing.
            remaining_budget = budget - calls_made
            
            if remaining_budget > 10:
                # Cap maxiter to ensure we don't blow the budget or time
                # Each L-BFGS-B iteration might take a few function calls.
                # Let's set a reasonable maxiter.
                max_iter_lbfgs = max(10, min(100, remaining_budget // 5))
                
                def lbfgs_func(x):
                    if time_check():
                        return 1e10
                    calls_made += 1
                    return func(x)
                
                try:
                    bounds = [(-5, 5)] * dim
                    res = minimize(lbfgs_func, best_x, method='L-BFGS-B', 
                                   bounds=bounds, 
                                   options={'maxiter': max_iter_lbfgs, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        # 3. If still have budget and time, do a final Nelder-Mead polish
        # Nelder-Mead is derivative-free and can refine further.
        if not time_check() and (budget - calls_made) > 10:
            remaining_calls = budget - calls_made
            max_iter_nm = max(5, min(50, remaining_calls // (dim + 1)))
            
            def nm_func(x):
                if time_check():
                    return 1e10
                calls_made += 1
                return func(x)
            
            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter_nm, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
