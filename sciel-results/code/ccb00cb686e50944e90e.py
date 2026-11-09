"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential Evolution based optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = float('inf')
        calls_used = 0
        max_calls = budget

        def wrapped_func(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= max_calls:
                return best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            val = float(func(x_clipped))
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Differential Evolution parameters
        # popsize is the number of individuals in the population
        # Each generation evaluates popsize new individuals
        # Initial population is also popsize
        # Total calls ~ popsize + maxiter * popsize
        
        # We want to use most of the budget.
        # Let's pick a popsize that allows for a reasonable number of generations.
        # dim=5. popsize=10 is a good start.
        popsize = 10
        if budget < popsize * 2:
            # Fallback to simple random search or single Nelder-Mead if budget is tiny
            # But budget is 500, so this won't happen.
            popsize = max(5, budget // 10)
            
        max_iter = (budget - popsize) // popsize
        if max_iter < 1:
            max_iter = 1
            popsize = budget - 1
            if popsize < 5:
                popsize = 5
                max_iter = 0 # Just do initial population

        bounds = [(-5.0, 5.0)] * dim
        
        try:
            # seed=None for randomness, but we can use a fixed seed for reproducibility if needed?
            # The task says "hidden seeded shift", implying the function is deterministic per instance.
            # Using a random seed for DE is fine.
            res = differential_evolution(
                wrapped_func,
                bounds,
                maxiter=max_iter,
                popsize=popsize,
                tol=1e-12,
                init='latinhypercube',
                strategy='best1bin',
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=None,
                polish=False # We handle polishing or skip it to save budget?
                             # polish uses Nelder-Mead which might use extra calls.
                             # If we set polish=False, we rely on DE's convergence.
                             # Let's try polish=True but be careful with budget.
                             # Actually, polish can use many calls. 
                             # Let's stick to polish=False to strictly control budget, 
                             # or implement a manual polish step.
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass

        # Optional: Manual polish with Nelder-Mead if budget remains
        remaining = max_calls - calls_used
        if remaining > 5 and best_x is not None:
            from scipy.optimize import minimize
            try:
                # Use a small maxfun to not overshoot
                polish_maxfun = min(remaining, 50)
                res2 = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': polish_maxfun, 'maxfev': polish_maxfun, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
