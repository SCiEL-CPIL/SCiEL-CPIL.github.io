import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        best_x = None
        best_f = np.inf
        calls = 0
        max_calls = self.budget
        
        def wrapped_func(x):
            nonlocal best_x, best_f, calls
            if time.time() > deadline:
                raise TimeoutError()
            
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        bounds = [(-5, 5)] * self.dim
        
        try:
            differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=49, 
                popsize=10, 
                seed=0, 
                polish=False, 
                tol=1e-12,
                init='sobol'
            )
        except TimeoutError:
            pass
        except Exception:
            pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            try:
                best_f = func(best_x)
            except:
                best_f = 0.0

        return best_x, best_f
