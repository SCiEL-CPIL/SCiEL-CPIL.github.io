"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer with strict budget enforcement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        d = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = None
        best_fx = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_fx
            call_count += 1
            if call_count > budget:
                raise ValueError("Budget exceeded")
            
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            
            if fx < best_fx:
                best_fx = fx
                best_x = np.array(x, dtype=float).copy()
                
            return fx

        # Generate 10 diverse initial points using a Halton-like sequence or fixed random
        rng = np.random.default_rng(42)
        # We can use a grid or random points. Random is fine with a good seed.
        # Let's use 10 points.
        num_starts = 10
        x0_candidates = []
        
        # Include the center and corners/edges to be safe
        x0_candidates.append(np.zeros(d))
        
        # Add random points
        for _ in range(num_starts - 1):
            x0 = rng.uniform(-4, 4, size=d)
            x0_candidates.append(x0)

        # Define bounds for L-BFGS-B
        bounds = [(-5.0, 5.0)] * d

        for x0 in x0_candidates:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break
                
            try:
                # maxiter is a soft limit, the call counter is the hard limit
                # L-BFGS-B is efficient, so a few iterations should suffice
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 50, # Limit iterations to prevent runaway
                        'ftol': 1e-10,
                        'gtol': 1e-8
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
                    
            except ValueError as e:
                if "Budget exceeded" in str(e):
                    break
                else:
                    raise e
            except Exception:
                # If optimization fails for other reasons, continue to next start
                continue

        if best_x is None:
             # Fallback if no calls were made or all failed
             best_x = np.zeros(d)
             try:
                 best_fx = wrapped_func(best_x)
             except:
                 best_fx = 0.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
