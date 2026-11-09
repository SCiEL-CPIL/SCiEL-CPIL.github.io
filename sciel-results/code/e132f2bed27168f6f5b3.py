"""
Strategy: Multi-Start Diagonal CMA-ES with Nelder-Mead Refinement.
Rationale: The Discus function (f11) is highly anisotropic. Diagonal CMA-ES adapts
per-dimension step sizes, allowing it to rapidly adapt to the ellipsoidal shape of
the landscape. Multi-starting helps escape the flat plateau and find the basin.
Once the search is close to the minimum, Nelder-Mead provides superior sample
efficiency for the final convergence.
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
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Helper to safely call func and track budget
        def safe_call(x):
            nonlocal total_calls, best_f, best_x
            if total_calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            total_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Diagonal CMA-ES Implementation ---
        def diag_cma_es(x0, sigma0, max_calls):
            """
            Diagonal CMA-ES with per-dimension sigma adaptation.
            Returns the best point found within the call budget.
            """
            nonlocal total_calls, best_f, best_x
            
            if total_calls >= budget:
                return best_x, best_f

            # Parameters
            lam = int(4 + 3 * np.log(dim))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            # Adaptation rates
            c_sigma = 1 + 2 * max(0, np.sqrt((lam - 2) / (dim + 1)) - 1) / (dim + 4)
            c_1 = (2 / ((dim + 1.3)**2 + mu_eff)) * (2 - c_sigma)
            c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)) * (c_sigma - 1 / (dim + 2)))
            chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            mean = x0.copy()
            sigmas = np.full(dim, sigma0)
            
            start_calls = total_calls
            stagnation = 0
            prev_best = best_f
            
            while total_calls - start_calls < max_calls and total_calls < budget:
                if time.time() > deadline - 5:
                    break
                    
                # Generate population
                ys = np.random.standard_normal((lam, dim))
                xs = mean + np.outer(np.ones(lam), sigmas) * ys
                xs = np.clip(xs, -5.0, 5.0)
                
                # Evaluate
                fs = np.full(lam, np.inf)
                for i in range(lam):
                    if total_calls >= budget:
                        break
                    f_val = safe_call(xs[i])
                    if f_val is None:
                        break
                    fs[i] = f_val
                    
                if total_calls >= budget:
                    break
                
                # Sort
                valid_indices = np.where(fs < np.inf)[0]
                if len(valid_indices) < mu:
                    break
                idx = np.argsort(fs[valid_indices])
                valid_indices = valid_indices[idx]
                
                # Update best
                if fs[valid_indices[0]] < best_f:
                    stagnation = 0
                else:
                    stagnation += 1
                    if stagnation > 3 * lam:
                        break # Stagnation
                
                # Update mean (recombination)
                # Use the top mu points
                top_xs = xs[valid_indices[:mu]]
                top_ys = ys[valid_indices[:mu]]
                top_fs = fs[valid_indices[:mu]]
                
                # Weighted mean of y's
                y_w = np.sum(w[:mu, np.newaxis] * top_ys, axis=0)
                new_mean = mean + sigmas * y_w
                new_mean = np.clip(new_mean, -5.0, 5.0)
                
                # Update sigmas (Diagonal CMA-ES update)
                # Rank-1 part
                rank_one = y_w**2
                # Rank-mu part
                rank_mu = np.sum(w[:mu, np.newaxis] * top_ys**2, axis=0)
                
                # Natural gradient step on log-sigmas
                # log(sigma_i) += (c_1 * rank_one_i + c_mu * rank_mu_i - 1) / (2 * mu_eff)
                log_sig_update = (c_1 * rank_one + c_mu * rank_mu - 1.0) / (2.0 * mu_eff)
                
                # Apply update with damping to prevent instability
                log_sigmas = np.log(sigmas)
                log_sigmas += log_sig_update
                log_sigmas = np.clip(log_sigmas, -10, 10)
                sigmas = np.exp(log_sigmas)
                
                # Ensure sigmas don't get too small
                sigmas = np.maximum(sigmas, 1e-12)
                
                mean = new_mean
                
                # Check convergence
                if np.min(sigmas) < 1e-10:
                    break
                    
            return best_x, best_f

        # --- Nelder-Mead Implementation ---
        def nelder_mead(x0, f0, max_calls):
            """
            Standard Nelder-Mead.
            """
            nonlocal total_calls, best_f, best_x
            
            if total_calls >= budget:
                return best_x, best_f
            
            # Initialize simplex
            simplex = [(x0.copy(), f0)]
            start_calls = total_calls
            
            # Expand simplex
            for i in range(dim):
                if total_calls >= budget or total_calls - start_calls >= max_calls:
                    break
                xi = x0.copy()
                step = 0.5
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append((xi, fi))
            
            if len(simplex) < dim + 1:
                return best_x, best_f
                
            simplex.sort(key=lambda t: t[1])
            
            alpha = 0.5
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while total_calls - start_calls < max_calls and total_calls < budget:
                if time.time() > deadline - 5:
                    break
                
                # Check convergence
                x_center = np.mean([s[0] for s in simplex], axis=0)
                spread = np.max(np.abs(np.array([s[0] for s in simplex]) - x_center))
                if spread < 1e-5:
                    break
                
                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_second_worst = simplex[-2][0]
                f_second_worst = simplex[-2][1]
                
                x_centroid = np.mean([s[0] for s in simplex[:-1]], axis=0)
                
                # Reflection
                x_reflect = x_centroid + gamma * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = safe_call(x_reflect)
                if f_reflect is None:
                    break
                
                if f_reflect < simplex[0][1]:
                    # Expansion
                    x_expand = x_centroid + alpha * (x_reflect - x_worst)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = safe_call(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = (x_expand, f_expand)
                    else:
                        simplex[-1] = (x_reflect, f_reflect)
                elif f_reflect < f_second_worst:
                    simplex[-1] = (x_reflect, f_reflect)
                else:
                    # Contraction
                    if f_reflect < f_worst:
                        x_contract = x_centroid + rho * (x_reflect - x_worst)
                    else:
                        x_contract = x_centroid - rho * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = safe_call(x_contract)
                    if f_contract is None:
                        break
                    if f_contract <= min(f_reflect, f_worst):
                        simplex[-1] = (x_contract, f_contract)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if total_calls >= budget:
                                break
                            new_x = simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])
                            new_x = np.clip(new_x, -5.0, 5.0)
                            new_f = safe_call(new_x)
                            if new_f is None:
                                break
                            simplex[i] = (new_x, new_f)
                        if total_calls >= budget:
                            break
                
                simplex.sort(key=lambda t: t[1])
                
            return best_x, best_f

        # --- Main Strategy ---
        
        # Phase 1: Multi-Start Diagonal CMA-ES
        # We divide the budget into multiple starts.
        # For dim=5, budget=500.
        # Let's try 3 starts: 100, 100, 100 calls for CMA-ES, leaving 200 for NM?
        # Or maybe 2 starts of 150 calls, leaving 200 for NM?
        # The inspiration 1 used multi-start NM. The current program used single CMA-ES.
        # Let's use 2 starts of CMA-ES.
        
        # Start 1: Center
        # Start 2: Random
        
        # Allocate 60% of budget to CMA-ES phases, 40% to NM.
        cma_total_budget = int(budget * 0.6)
        nm_total_budget = budget - cma_total_budget
        
        # Split CMA-ES budget into 2 starts
        cma_budget_per_start = cma_total_budget // 2
        
        # Start 1
        x0_1 = np.zeros(dim)
        sigma0_1 = 2.0
        diag_cma_es(x0_1, sigma0_1, cma_budget_per_start)
        
        # Start 2
        rng = np.random.default_rng(42)
        x0_2 = rng.uniform(-4.0, 4.0, dim)
        sigma0_2 = 2.0
        diag_cma_es(x0_2, sigma0_2, cma_budget_per_start)
        
        # Phase 2: Nelder-Mead Refinement
        remaining = budget - total_calls
        if remaining > 10:
            nelder_mead(best_x, best_f, remaining)

        return best_x, best_f
