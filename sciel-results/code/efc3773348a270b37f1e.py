import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = (-5, 5)
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        # Wrapper to count calls and track best
        def wrapper(x):
            if time.time() > deadline:
                raise TimeoutError()
            if self.calls >= self.budget:
                # Return a large value to stop optimization or handle gracefully
                # Actually, if we exceed budget, we should stop.
                # But differential_evolution might not stop immediately.
                # We will raise an exception to break out.
                raise BudgetExceeded()
            
            self.calls += 1
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val
        
        # Phase 1: Global Optimization with Differential Evolution
        # Use ~80% of budget
        popsize = 10
        # maxiter such that popsize * (1 + maxiter) <= 0.8 * budget
        maxiter = int((0.8 * self.budget / popsize) - 1)
        
        try:
            res = differential_evolution(
                wrapper, 
                bounds=[self.bound] * self.dim, 
                popsize=popsize, 
                maxiter=maxiter, 
                tol=1e-4, 
                seed=42,
                polish=False # We will polish manually
            )
            # Update best from result
            if res.fun < self.best_f:
                self.best_f = res.fun
                self.best_x = res.x.copy()
        except Exception:
            pass # If DE fails, proceed with best found so far

        # Phase 2: Local Refinement with Nelder-Mead
        # Use remaining budget
        if self.best_x is not None and self.calls < self.budget and time.time() < deadline:
            try:
                # Nelder-Mead can be sensitive to initial simplex size, 
                # but starting from a good DE point is usually fine.
                # maxiter for minimize is number of iterations, not evals.
                # Roughly 10-20 evals per iteration for NM in 5D? 
                # Let's just run it and let the wrapper stop it when budget is hit.
                res_local = minimize(
                    wrapper, 
                    self.best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res_local.fun < self.best_f:
                    self.best_f = res_local.fun
                    self.best_x = res_local.x.copy()
            except Exception:
                pass

        # Ensure we return valid types
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(self.best_x)
            self.calls += 1

        return self.best_x, self.best_f

class BudgetExceeded(Exception):
    pass
