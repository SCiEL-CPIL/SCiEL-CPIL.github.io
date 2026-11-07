import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0  # Safety margin before 300s limit
        d = self.dim
        
        # State tracking
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def time_ok():
            return (time.time() - start_time) < timeout

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                raise ValueError("Budget exceeded")
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Differential Evolution for Global Search
        # We allocate 80% of the budget to DE
        de_budget = int(self.budget * 0.8)
        popsize = 20
        maxiter = max(1, de_budget // popsize)
        
        bounds = [(-5.0, 5.0)] * d
        
        try:
            # DE is a global optimizer, good for finding the step region
            res_de = scipy.optimize.differential_evolution(
                safe_func,
                bounds,
                popsize=popsize,
                maxiter=maxiter,
                tol=1e-8,
                seed=42,
                polish=False # Disable internal polish to save calls for our local search
            )
            # Update best if DE found something better (it should, as safe_func updates it)
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x.copy()
        except Exception:
            # If DE fails, best_x might still be set from partial runs
            pass

        # Fallback: If no calls were made or best_x is None, evaluate center
        if best_x is None:
            x_center = np.zeros(d)
            try:
                fx = safe_func(x_center)
                best_x = x_center.copy()
                best_f = fx
            except Exception:
                pass

        # Phase 2: Local Refinement with Powell's Method
        # Powell is robust for local search
        if best_x is not None and calls_made < self.budget and time_ok():
            remaining_budget = self.budget - calls_made
            # Powell's method call count is not strictly linear, but we can limit maxiter
            # Each iteration in Powell involves evaluating along each coordinate axis
            # So roughly d evaluations per iteration.
            maxiter_powell = max(1, remaining_budget // (d + 1))
            
            try:
                res_powell = scipy.optimize.minimize(
                    safe_func,
                    best_x,
                    method='Powell',
                    options={
                        'xtol': 1e-8,
                        'ftol': 1e-10,
                        'maxiter': maxiter_powell,
                        'disp': False
                    }
                )
                if res_powell.fun < best_f:
                    best_f = res_powell.fun
                    best_x = res_powell.x.copy()
            except Exception:
                pass

        return best_x, best_f
