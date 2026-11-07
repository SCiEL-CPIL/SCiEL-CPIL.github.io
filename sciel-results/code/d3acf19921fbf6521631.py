"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Scout + L-BFGS-B/Nelder-Mead for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 330s limit
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = np.clip(x, -5, 5)
            
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # Phase 1: Global Scout
        # Use a mix of deterministic and random points to locate the basin.
        # The minimum is in [-4, 4]^5.
        # We will use 20 random points and a few deterministic ones.
        
        scout_points = []
        
        # Deterministic
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))
        
        # Random points in the box
        # Ensure we don't exceed budget
        max_scout = min(30, budget // 10) # Cap scout phase to 10% of budget, max 30
        num_random = max(0, max_scout - len(scout_points))
        
        for _ in range(num_random):
            scout_points.append(np.random.uniform(-5, 5, dim))
            
        scored_candidates = []
        for p in scout_points:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                fp = wrapped_func(p)
                if np.isfinite(fp):
                    scored_candidates.append((fp, p.copy()))
            except Exception:
                break

        if best_x is None:
             # Fallback
             best_x = np.zeros(dim)
             try:
                 best_f = func(best_x)
                 eval_count += 1
             except Exception:
                 pass

        # Sort candidates by function value
        scored_candidates.sort(key=lambda x: x[0])

        # Phase 2: Local Optimization with L-BFGS-B
        # L-BFGS-B is efficient for smooth functions and exploits Rosenbrock's structure.
        # We run it from the top few candidates.
        
        max_local_starts = min(5, len(scored_candidates))
        
        for i in range(max_local_starts):
            if time.time() > deadline or eval_count >= budget:
                break
            
            fx0, x0 = scored_candidates[i]
            
            try:
                remaining = budget - eval_count - 10
                if remaining <= 0:
                    break
                
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'xtol': 1e-8, 
                        'ftol': 1e-10,
                        'maxiter': 100,
                        'maxfun': remaining
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # Phase 3: Refinement with Nelder-Mead
        # Nelder-Mead can polish the result if L-BFGS-B stopped early or didn't converge fully.
        if best_x is not None and eval_count < budget and time.time() < deadline:
            x0 = best_x.copy()
            
            try:
                remaining = budget - eval_count
                if remaining > 0:
                    res = minimize(
                        wrapped_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'xatol': 1e-6, 
                            'fatol': 1e-8,
                            'maxiter': 100,
                            'maxfev': remaining
                        }
                    )
                    
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
            except Exception:
                pass

        if best_x is None:
             best_x = np.zeros(dim)
             try:
                 best_f = func(best_x)
             except Exception:
                 pass

        return best_x, best_f
# EVOLVE-BLOCK-END
