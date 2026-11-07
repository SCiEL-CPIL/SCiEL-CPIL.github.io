"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start local search using scipy minimize (Nelder-Mead)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0  # safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # We have 500 function evaluations total.
        # Each Nelder-Mead call in 5D uses roughly (dim+1) = 6 initial points
        # plus iterations. A typical Nelder-Mead run might use 50-200 evals.
        # Let's do multiple restarts.
        # Strategy: Use a fixed number of restarts, but monitor time.
        # If we run out of time, we stop.
        # We need to track function calls to not exceed 500.
        
        # Since we can't easily count calls inside scipy.optimize.minimize 
        # without wrapping, and the budget is strict (500), 
        # we should be careful.
        # However, the task says "More than 500 calls ... makes the whole result invalid".
        # So we must strictly limit the total calls to <= 500.
        
        # scipy.optimize.minimize does not have a strict "max_calls" for the total 
        # across multiple restarts, but we can wrap the function to count.
        
        call_count = [0]
        max_calls = self.budget  # 500

        def counted_func(x):
            if call_count[0] >= max_calls:
                return best_f  # Return current best to avoid error, though we should stop
            call_count[0] += 1
            return func(x)

        # We will perform a series of local optimizations from different starts.
        # How many? 
        # If each local opt takes ~100 calls, we can do ~4-5 restarts.
        # If each takes ~20 calls, we can do ~20 restarts.
        # Nelder-Mead in 5D is fast. Let's aim for ~10-20 restarts.
        
        # To be safe and ensure we use the budget well:
        # 1. Start with a few random points.
        # 2. Run minimize from each.
        # 3. If time or calls run out, stop.
        
        # Let's try 10 restarts.
        n_restarts = 10
        
        for i in range(n_restarts):
            if not time_ok() or call_count[0] >= max_calls:
                break
            
            # Random start in [-5, 5]^5
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Check if we have enough budget for a reasonable local search
            # Nelder-Mead options: maxiter, maxfev
            # Let's set maxfev to a reasonable chunk of remaining budget
            remaining = max_calls - call_count[0]
            # Don't use all remaining in one go, leave some for others?
            # Or just let it run until it converges or hits maxfev.
            # Since we have a global time limit, we can be generous with maxfev 
            # and rely on the time guard, BUT we must not exceed 500 total calls.
            
            # If we set maxfev=remaining, and it doesn't converge, it will use all.
            # If it converges early, it uses less.
            # This is a good strategy: use as much as needed up to the limit.
            
            try:
                res = minimize(
                    counted_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If minimize fails, continue to next restart
                pass

        # If we still have budget and time, we could do a final polish?
        # Nelder-Mead with adaptive=True is usually good.
        # Let's check if we have significant time left and calls left.
        if time_ok() and call_count[0] < max_calls and best_x is not None:
            # Do one more polish from the best point found
            remaining = max_calls - call_count[0]
            if remaining > 10: # Need some calls
                try:
                    res = minimize(
                        counted_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxfev': remaining, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        # Return the best found
        if best_x is None:
            # Fallback if something went wrong
            x0 = np.zeros(self.dim)
            best_f = func(x0)
            best_x = x0
            
        return best_x, best_f
# EVOLVE-BLOCK-END
