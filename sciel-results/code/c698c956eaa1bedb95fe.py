"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with Nelder-Mead polish for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, total_calls
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: (1+1)-ES with adaptive restarts
        # Start at center with moderate sigma
        x = np.zeros(n)
        f_x = evaluate(x)
        sigma = 2.0
        last_improve_call = total_calls

        remaining_calls = budget - total_calls

        while remaining_calls > 0:
            if time.time() > deadline:
                break

            # Check for stagnation: if sigma is very small and no improvement recently
            if sigma < 1e-4 and (total_calls - last_improve_call) > 50:
                # Try to escape: kick from best with larger sigma
                x = best_x + np.random.standard_normal(n) * 1.0
                x = np.clip(x, -5, 5)
                f_x = evaluate(x)
                remaining_calls -= 1
                if f_x < best_f:
                    last_improve_call = total_calls
                else:
                    # Reset to best and try with larger sigma
                    x = best_x.copy()
                    f_x = best_f
                    sigma = 0.5
                continue

            # Generate offspring
            z = np.random.standard_normal(n)
            y = x + sigma * z
            y = np.clip(y, -5, 5)

            f_y = evaluate(y)
            remaining_calls -= 1

            if f_y < f_x:
                x = y
                f_x = f_y
                sigma *= 1.05
                last_improve_call = total_calls
            else:
                sigma *= 0.95

            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # If we have very few calls left, reduce sigma to ensure we don't jump out of the basin
            if remaining_calls < 20:
                sigma = min(sigma, 0.01)

        # Phase 2: Nelder-Mead polish from the best point found
        # Use the remaining budget for local refinement
        if best_x is not None and remaining_calls > 10:
            # Allocate up to 50% of remaining budget for polish, but at least 10 calls
            polish_budget = min(remaining_calls, max(10, remaining_calls // 2))
            
            # Use Nelder-Mead with a tight tolerance
            # We need to track calls carefully
            polish_calls = [0]
            
            def polish_func(x):
                polish_calls[0] += 1
                return func(x)
            
            try:
                # Start from the best point found by ES
                x0 = best_x.copy()
                
                # Run Nelder-Mead with limited iterations
                # maxiter is not a direct call count, so we use a callback to stop
                result = minimize(
                    polish_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    },
                    callback=lambda x: None if polish_calls[0] < polish_budget else (_ for _ in ()).throw(Exception("stop"))
                )
                
                # Update best if Nelder-Mead found something better
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                # If Nelder-Mead raises an exception (e.g., from our stop mechanism),
                # we just continue with what we have
                pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
