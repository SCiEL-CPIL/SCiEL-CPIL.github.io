import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer using a Hybrid Global-Local strategy.
    1. Global Exploration: Latin Hypercube Sampling to identify promising basins.
    2. Local Optimization: Multi-stage Nelder-Mead + Powell polish from top candidates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 290s to be safe
        deadline = start_time + 290
        
        n = self.dim
        total_budget = self.budget
        
        calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        # Helper to clip to box
        def clip(x):
            return np.clip(x, -5, 5)

        # Evaluate function wrapper
        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= total_budget or time.time() > deadline:
                return np.inf
            x = clip(x)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        # Use 25% of budget for exploration, clamped to [50, 125]
        n_explore = int(total_budget * 0.25)
        n_explore = max(50, min(125, n_explore))
        
        if n_explore >= total_budget:
            # If budget is very small, just use random
            n_explore = total_budget
            points = np.random.uniform(-5, 5, (n_explore, n))
        else:
            # Generate Latin Hypercube samples
            sampler = qmc.LatinHypercube(d=n, seed=42)
            samples = sampler.random(n_explore)
            points = -5 + 10 * samples # Scale to [-5, 5]
        
        exploration_results = []
        
        for p in points:
            if calls >= total_budget or time.time() > deadline:
                break
            fx = eval_func(p)
            if not np.isinf(fx):
                exploration_results.append((fx, p))
            else:
                # If fx is inf, it's likely a bad point or error, but we still record the call
                exploration_results.append((np.inf, p))

        # Phase 2: Local Optimization
        # Sort by fitness to find the best starting points
        exploration_results.sort(key=lambda x: x[0])
        
        # Select top K candidates
        # We have total_budget - calls evaluations left
        n_remaining = total_budget - calls
        
        if n_remaining > 0 and exploration_results:
            # Choose K such that we have at least 25 evals per local search
            # But not more than 8 starts
            K = min(8, max(1, n_remaining // 35))
            
            # Ensure K is valid
            K = max(1, min(K, n_remaining // 25))
            if K < 1:
                K = 1
                
            top_candidates = [p for _, p in exploration_results[:K]]
            
            budget_per_search = n_remaining // K
            
            for start_p in top_candidates:
                if calls >= total_budget or time.time() > deadline:
                    break
                
                # Wrapper for local search to enforce strict budget
                def local_func(x):
                    if calls >= total_budget or time.time() > deadline:
                        raise ValueError("Budget Exceeded")
                    return eval_func(x)
                
                # Stage 1: Nelder-Mead for quick convergence
                try:
                    maxiter_nm = max(1, budget_per_search // 6)
                    res_nm = minimize(local_func, start_p, method='Nelder-Mead', 
                                       options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                    current_x = res_nm.x
                except ValueError:
                    break
                except Exception:
                    current_x = start_p
                
                # Stage 2: Powell polish with tighter tolerances if budget remains
                if calls < total_budget and time.time() < deadline:
                    remaining_for_polish = total_budget - calls
                    if remaining_for_polish > 10:
                        try:
                            maxiter_pw = max(1, remaining_for_polish // 8)
                            res_pw = minimize(local_func, current_x, method='Powell',
                                              options={'maxiter': maxiter_pw, 'xtol': 1e-12, 'ftol': 1e-12})
                        except ValueError:
                            break
                        except Exception:
                            pass
        
        return best_x, best_f
