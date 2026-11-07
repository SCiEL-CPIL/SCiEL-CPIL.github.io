import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = None
        self.deadline = 300.0 # 300s limit, use 290s for safety

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0
        
        # Wrapper for function to track calls and time
        def wrapped_func(x):
            self.call_count += 1
            if self.call_count > self.budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            return func(x)

        # Initial points
        # The minimizer is in [-4, 4]^5.
        # Try a few diverse starts.
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim),
            np.full(self.dim, -1.0),
            np.random.uniform(-4, 4, self.dim),
            np.random.uniform(-4, 4, self.dim)
        ]
        
        best_x = None
        best_f = np.inf

        # Phase 1: Global-ish search with Powell
        # Powell is derivative-free and good for Rosenbrock
        for start in starts:
            if self.call_count >= self.budget - 50: # Leave room for polish
                break
            if time.time() > self.deadline - 30:
                break
                
            try:
                # Use Powell's method
                # maxiter is not strictly calls, but we can limit by wrapping
                # SciPy's Powell doesn't have a direct 'maxfev' in all versions? 
                # Actually, scipy.optimize.minimize has options.
                # For Powell, 'maxiter' is the number of iterations.
                # We can estimate calls per iteration ~ dim + 1.
                # Let's just run it and let the wrapper stop it if budget/time runs out.
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    start, 
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Phase 2: Local Polish with L-BFGS-B
        # L-BFGS-B uses finite differences for gradient.
        # Cost per iteration ~ 2*dim calls.
        # With 5D, ~10 calls/iter.
        # If we have 100 calls left, ~10 iterations.
        if best_x is not None and self.call_count < self.budget - 10:
            try:
                res2 = scipy.optimize.minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x
            except Exception:
                pass

        return best_x, best_f
