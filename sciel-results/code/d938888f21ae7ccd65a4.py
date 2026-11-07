import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285
        rng = np.random.default_rng(42)
        
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1
        
        # Phase 1: Coarse random search to find a good starting point
        # Use a fraction of the budget for exploration
        explore_budget = min(50, budget // 5)
        for i in range(explore_budget):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, size=dim)
            fx = func(x)
            calls += 1
            if fx < fx_best:
                x_best = x.copy()
                fx_best = fx
        
        # Phase 2: Powell's method for refinement
        # Powell is effective for Rosenbrock-like curved valleys
        # We use multiple restarts with different initial points near the best found
        remaining_budget = budget - calls
        if remaining_budget > 0:
            # Try Powell from the best point found
            try:
                result = minimize(
                    func, x_best, method='Powell',
                    options={'maxiter': remaining_budget, 'maxfev': remaining_budget}
                )
                if result.fun < fx_best:
                    x_best = result.x.copy()
                    fx_best = result.fun
                    calls += result.nfev
            except Exception:
                pass
            
            # If we still have budget, try a few more local optimizations
            # from nearby points to escape potential local minima
            remaining_budget = budget - calls
            if remaining_budget > 10:
                for restart in range(min(3, remaining_budget // 20)):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Generate a point near the current best
                    perturbation = 0.5 * rng.standard_normal(dim)
                    x_start = np.clip(x_best + perturbation, -5, 5)
                    fx_start = func(x_start)
                    calls += 1
                    
                    if fx_start < fx_best:
                        x_best = x_start.copy()
                        fx_best = fx_start
                    
                    try:
                        result = minimize(
                            func, x_start, method='Powell',
                            options={'maxiter': remaining_budget // 2, 'maxfev': remaining_budget // 2}
                        )
                        if result.fun < fx_best:
                            x_best = result.x.copy()
                            fx_best = result.fun
                            calls += result.nfev
                        else:
                            calls += result.nfev
                    except Exception:
                        pass
        
        return x_best, fx_best
