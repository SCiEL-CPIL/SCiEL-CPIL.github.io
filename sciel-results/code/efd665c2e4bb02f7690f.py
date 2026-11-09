import numpy as np
import time
from scipy.stats.qmc import Sobol
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Wrapper to track calls, time, and best result
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls_made >= self.budget:
                raise Exception("Budget exceeded")
            
            calls_made += 1
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # Phase 1: Global Exploration with Sobol Sequence
        # Use 150 points for global search in 5D
        n_global = min(150, self.budget)
        
        try:
            # Generate Sobol points
            sampler = Sobol(d=self.dim, scramble=True)
            # Generate n_global points in [0, 1]^d
            sobol_points = sampler.random(n_global)
            # Map to [-5, 5]^d
            sobol_points = sobol_points * 10.0 - 5.0
            
            for x in sobol_points:
                wrapped_func(x)
                
        except (TimeoutError, Exception):
            pass

        # If we haven't made any calls (should not happen), return
        if best_x is None:
             # Fallback: try center
             x0 = np.zeros(self.dim)
             try:
                 wrapped_func(x0)
             except:
                 pass
             if best_x is None:
                 return np.zeros(self.dim), np.inf

        # Phase 2: Local Refinement with Nelder-Mead
        # Use remaining budget
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 0:
            try:
                # Nelder-Mead options
                # xatol and fatol control convergence
                # maxiter is high, but budget/time guards will stop it
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'maxiter': 100000,
                        'adaptive': True
                    }
                )
            except (TimeoutError, Exception):
                pass

        return best_x, best_f
