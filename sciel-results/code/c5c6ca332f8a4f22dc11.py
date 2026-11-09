import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls_used = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Wrapper to count calls and handle deadline
        def wrapped_func(x):
            nonlocal calls_used
            if time.time() > deadline or calls_used >= budget:
                return best_f # Return current best to signal stop? No, this might confuse optimizer.
                # Better: raise an exception or return a large value?
                # If we return best_f, the optimizer might think it's stuck.
                # If we raise, we catch it outside.
                # Let's just check before calling.
            calls_used += 1
            return func(x)

        # Strategy: Multi-start local search
        # 1. Initial random sampling to find a few good basins
        n_initial = 10
        candidates = []
        for i in range(n_initial):
            if calls_used >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped_func(x)
            calls_used += 1 # wrapped_func already incremented? No, I defined it to increment.
            # Wait, wrapped_func increments calls_used.
            # So fx = wrapped_func(x) increments calls_used.
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        
        # 2. Run local optimization from the best candidates
        # Powell's method is robust.
        # We allocate remaining budget to these runs.
        # Each run gets a limited number of iterations.
        # In 5D, Powell takes ~20-50 iterations to converge.
        # Each iteration is ~5-10 function calls.
        # So ~200 calls per run is safe.
        # We have ~490 calls left. We can run ~2-3 good local searches.
        
        num_local_runs = 3
        for i in range(num_local_runs):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Pick the i-th best candidate, or a random one if i >= len(candidates)
            if i < len(candidates):
                x0 = candidates[i][1]
            else:
                x0 = rng.uniform(-5, 5, dim)
            
            # Run Powell
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={'maxiter': 50, 'maxfev': 100} # Limit function evals per run
                )
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass
            
            # If we found a better point, we might want to restart from it?
            # Powell usually converges to a local min.
            
        # 3. If budget remains, do more random starts + local refinement
        while calls_used < budget - 50 and time.time() < deadline:
            x0 = rng.uniform(-5, 5, dim)
            fx0 = wrapped_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            
            # Quick local refinement with Nelder-Mead (faster per iter?)
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 20, 'maxfev': 50}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return (best_x, best_f)
