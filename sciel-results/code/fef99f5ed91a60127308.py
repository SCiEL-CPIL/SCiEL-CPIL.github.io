"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Robust (1+1)-ES followed by Nelder-Mead polish.
    Strict budget tracking to avoid exceeding 500 calls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        calls_made = 0
        max_calls = self.budget
        # Reserve some calls for polish phase
        polish_budget = 50
        es_budget = max_calls - polish_budget

        # Initial point: center or random
        # Since there is a shift, random is safer, but center is good if shift is small.
        # Let's use a few random starts and pick the best, or just one good random start.
        # With 500 calls, we can afford 5 initial points (5 calls) + 1 ES run.
        
        # Strategy: 
        # 1. Try 5 random starting points (5 calls).
        # 2. Run (1+1)-ES from the best starting point with es_budget - 5 calls.
        # 3. Run Nelder-Mead from the best ES result with polish_budget calls.

        # Step 1: Multi-start initialization
        n_starts = 5
        best_x = None
        best_f = np.inf
        
        for _ in range(n_starts):
            if calls_made >= max_calls:
                break
            x0 = rng.uniform(-4, 4, self.dim)
            fx = func(x0)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
            
            if time.time() > deadline:
                break

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1

        # Step 2: (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        
        # Target number of ES steps
        es_steps = es_budget - calls_made
        
        for i in range(es_steps):
            if calls_made >= max_calls:
                break
            if time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            # Clip to box
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.2
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9
                
            # Clamp sigma
            if sigma < 0.01:
                sigma = 0.01
            elif sigma > 5.0:
                sigma = 5.0

        # Step 3: Nelder-Mead Polish
        remaining_calls = max_calls - calls_made
        if remaining_calls > 10 and best_x is not None:
            # Wrap func to count calls and enforce deadline
            def wrapped_func(x_vec):
                nonlocal calls_made
                if calls_made >= max_calls or time.time() > deadline:
                    return 1e100 # Return large value to stop optimization
                val = func(x_vec)
                calls_made += 1
                return val

            try:
                # Nelder-Mead parameters
                # maxiter is not direct call count, but a limit on iterations.
                # Each iteration makes ~dim+1 calls.
                # We have remaining_calls. So maxiter ~ remaining_calls / (dim + 1)
                max_iter = max(1, remaining_calls // (self.dim + 1))
                
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass # Keep best_x if NM fails

        return best_x, best_f
# EVOLVE-BLOCK-END
