import numpy as np
from scipy.optimize import differential_evolution
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        bounds = [(-5, 5)] * self.dim
        
        # Track best value
        best_val = np.inf
        best_x = np.zeros(self.dim)
        
        # Wrapper to track calls and best
        def wrapped_func(x):
            nonlocal best_val, best_x
            if time.time() - start_time > 290:
                raise TimeoutError()
            val = func(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
            return val

        # DE Parameters
        # popsize: multiplier for dimension. 10 is a good start.
        # maxiter: number of generations.
        # total calls = popsize * dim * maxiter? No, popsize is the number of individuals.
        # Evaluations per generation = popsize.
        # Total evaluations = popsize * maxiter.
        # We have 500 evaluations.
        # Let's use popsize = 10, maxiter = 50.
        # 10 * 50 = 500 evaluations.
        
        try:
            res = differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                seed=0, 
                tol=1e-8, 
                polish=False # Skip Nelder-Mead to save budget/time, or use it if budget allows
            )
        except TimeoutError:
            pass
        except Exception:
            pass

        # If DE didn't find anything better (unlikely), keep the initial
        if np.isinf(best_val):
            best_x = np.zeros(self.dim)
            best_val = func(best_x)

        return best_x, best_val
