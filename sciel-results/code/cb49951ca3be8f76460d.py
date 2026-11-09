import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    """
    Sobol quasi-random global search followed by adaptive Nelder-Mead local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        calls = 0
        best_x = None
        best_f = np.inf
        all_points = []
        all_fvals = []

        def eval_pt(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            all_points.append(x.copy())
            all_fvals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol quasi-random global exploration
        # Allocate 50% of the budget to global search
        n_global = budget // 2
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        init_points = sampler.random(n=n_global)
        init_points = 2.0 * init_points - 1.0
        init_points = 5.0 * init_points
        
        for i in range(n_global):
            if calls >= budget or time.time() > deadline:
                break
            eval_pt(init_points[i])
            
        if best_x is None:
            best_x = np.zeros(dim)
            eval_pt(best_x)

        # Phase 2: Adaptive Local Refinement (Nelder-Mead)
        # Sort candidates by their global search fitness
        order = np.argsort(all_fvals)
        
        # We will attempt local optimization on the best candidates
        # until the budget is exhausted or we have tried several good starts
        max_local_attempts = 10
        attempts = 0
        
        # Track the best value before local search to detect improvement
        best_f_before_local = best_f
        
        for idx in order[:max_local_attempts]:
            if calls >= budget or time.time() > deadline:
                break
            
            start_x = all_points[idx].copy()
            
            # If this point is not better than the current global best, skip
            # (It might be equal due to floating point, but usually strictly better or equal)
            if all_fvals[idx] > best_f:
                continue

            # Define a wrapper for minimize that respects global constraints
            # We pass the current best_f to the wrapper to allow early stopping if 
            # we can't beat it? No, NM needs to explore.
            
            def nm_func(x):
                if calls >= budget or time.time() > deadline:
                    return best_f
                return eval_pt(x)

            options = {
                'maxiter': 50,  # Limit iterations to force diversity in restarts
                'xatol': 1e-5, 
                'fatol': 1e-7,
                'adaptive': True
            }
            
            try:
                res = minimize(nm_func, start_x, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass
            
            attempts += 1
            # If we have significantly improved, we might want to restart NM from the new best
            # but the loop structure handles the next best global point.
            # If the local search converged to the global best, subsequent NM runs on 
            # worse global points are likely to fail or be slow.
            # Heuristic: If we found a very good solution (e.g. < 1.0 for this function scale?), 
            # we could stop. But we don't know the scale.
            
        return best_x, best_f
