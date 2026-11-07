"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Structured Multi-Start Optimizer for Rastrigin (rotated).
    
    Strategy:
    1. Global phase: Sobol quasi-random points to cover the domain well.
    2. Identify top candidates.
    3. Local phase: Nelder-Mead refinement from top candidates.
    4. Budget management: explicit call counter and time guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_made = [0]
        best_f = np.inf
        best_x = None

        def counted_func(x):
            calls_made[0] += 1
            val = func(x)
            nonlocal best_f, best_x
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Sobol sampling
        # Use ~30% of budget for global exploration
        n_global = max(15, budget // 3)
        
        sampler = qmc.Sobol(d=dim, scramble=True, seed=12345)
        sample = sampler.random(n_global)
        # Scale to [-5, 5]^dim
        x_starts = sample * 10.0 - 5.0
        
        # Evaluate global points
        for i in range(n_global):
            if time.time() > deadline - 20 or calls_made[0] >= budget:
                break
            x = x_starts[i]
            counted_func(x)
        
        # Collect top K candidates from the global phase
        # We need to track the values. Let's re-evaluate or store.
        # To save calls, we can just pick the first few best from a stored list, 
        # but we didn't store all. Let's do a small local scan or just use the best few found?
        # Better: Run a second small Sobol set with different seed and combine, or just use top K from stored.
        # Let's modify counted_func to store top K? No, simple is better.
        
        # Let's re-structure: store all global results in a list
        # But we already consumed calls. Let's just pick the best X found so far?
        # We only have best_x. 
        # Let's run Nelder-Mead from best_x and a few random points near it.
        # Or, better, run Nelder-Mead from the best 3 points found in global phase.
        # To get 3 best points without storing all, we can do a mini global search?
        # No, let's just use the best point and 2-3 random perturbations of it.
        
        # Actually, let's restart with proper storage to be safe and clean.
        # I will rewrite the __call__ body cleanly.

    # Let's rewrite __call__ completely to be robust.
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_f = np.inf
        best_x = None
        all_results = []  # List of (f, x)

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            calls_made += 1
            val = float(func(x))
            all_results.append((val, x.copy()))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # 1. Global Sobol Search
        n_global = max(20, budget // 4)  # 25% of budget
        sampler = qmc.Sobol(d=dim, scramble=True, seed=99)
        sample = sampler.random(n_global)
        x_global = sample * 10.0 - 5.0
        
        for x in x_global:
            if time.time() > deadline - 30 or calls_made >= budget:
                break
            counted_func(x)
        
        if calls_made == 0:
            # At least try one point
            counted_func(np.zeros(dim))
            return best_x, best_f

        # 2. Select top K candidates for local refinement
        # Sort by function value
        all_results.sort(key=lambda t: t[0])
        top_k = min(5, len(all_results))
        candidates = [x for _, x in all_results[:top_k]]
        
        # 3. Local Refinement with Nelder-Mead
        # Distribute remaining budget among candidates
        remaining = budget - calls_made
        if remaining > 10:
            n_local_runs = min(top_k, 5)
            budget_per_run = remaining // n_local_runs
            budget_per_run = max(20, budget_per_run)  # At least 20 calls per run
            
            for i, x_start in enumerate(candidates[:n_local_runs]):
                if time.time() > deadline - 10 or calls_made >= budget:
                    break
                
                # Use a wrapper that stops Nelder-Mead when budget is exceeded
                local_calls = [0]
                max_local_calls = budget_per_run
                
                def local_func(x):
                    nonlocal local_calls
                    if local_calls[0] >= max_local_calls or calls_made >= budget or time.time() > deadline:
                        raise ValueError("Budget exceeded")
                    local_calls[0] += 1
                    val = counted_func(x)
                    return val

                try:
                    # Nelder-Mead parameters
                    res = minimize(local_func, x_start, method='Nelder-Mead',
                                   options={
                                       'maxiter': max_local_calls * 2,  # Loose limit, we stop via exception
                                       'xatol': 1e-4,
                                       'fatol': 1e-6,
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except ValueError:
                    pass  # Budget exceeded, continue
                except Exception:
                    pass  # Any other error, continue

        return best_x, best_f

# EVOLVE-BLOCK-END
