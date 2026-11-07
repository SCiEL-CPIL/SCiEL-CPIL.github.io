"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid optimizer: Random exploration + L-BFGS-B local refinement + (1+1)-ES polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        # Phase 1: Random Exploration to find diverse seeds
        # We spend a small portion of the budget on random sampling to identify promising basins.
        n_explore = 10
        seeds = []
        
        for i in range(n_explore):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            seeds.append((x.copy(), fx))
            
            # If we run out of budget during exploration, stop
            if calls_made >= budget:
                break

        if not seeds:
            # Fallback if we couldn't even make one call (should not happen)
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return best_x, best_f

        # Sort seeds by fitness to pick the best few for local optimization
        seeds.sort(key=lambda s: s[1])
        
        # Phase 2: L-BFGS-B Local Optimization from best seeds
        # We will run L-BFGS-B on the top 3 seeds. 
        # Each run is limited to a small number of function calls to stay within budget.
        # L-BFGS-B is very efficient for smooth functions like Rosenbrock.
        n_local_runs = 3
        calls_per_local = max(10, (budget - calls_made) // (n_local_runs + 1)) # Leave some budget for polishing
        
        for i in range(min(n_local_runs, len(seeds))):
            if not time_ok() or calls_made >= budget:
                break
            
            start_x, start_f = seeds[i]
            
            # Define wrapper for scipy
            def objective(x):
                nonlocal calls_made
                if not time_ok() or calls_made >= budget:
                    return 1e10 # Return large value to stop optimization
                fx = func(x)
                calls_made += 1
                try_update(x, fx)
                return fx

            # Run L-BFGS-B
            try:
                res = minimize(objective, start_x, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 50, 'maxfun': calls_per_local})
                # Update best if L-BFGS-B found something better
                if res.fun < best_f:
                    best_x = res.x.copy()
                    best_f = res.fun
            except Exception:
                pass # Ignore errors in local opt, continue to next seed

        # Phase 3: Polishing with (1+1)-ES
        # Use remaining budget for fine-grained search around the best point found.
        if best_x is not None and calls_made < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1 # Start with small sigma for polishing
            stall = 0
            max_stall = 20
            
            while calls_made < budget and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.1, 1.0)
                    stall = 0
                    try_update(x, fx)
                else:
                    sigma = max(sigma * 0.9, 1e-6)
                    stall += 1
                    if stall > max_stall:
                        break # Stop polishing if stuck

        return best_x, best_f
# EVOLVE-BLOCK-END
