import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        dim = self.dim
        budget = self.budget

        # 1. Initialization with Halton sequence for better spread
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q = q / base
                r = r + q * (index % base)
                index = index // base
            return r

        points = []
        # Use a larger initial spread to cover the domain better
        # The problem is in [-5, 5]^5, but minimizer is in [-4, 4]^5
        # Let's sample from [-5, 5] to be safe.
        n_init = 20
        for i in range(1, n_init + 1):
            p = np.array([halton_sequence(i, base) for base in range(2, 2 + dim)])
            p = p * 10.0 - 5.0
            points.append(p)

        # Add origin and some structured points
        points.append(np.zeros(dim))
        points.append(np.ones(dim) * 2.0)
        points.append(np.ones(dim) * -2.0)

        # Evaluate initial points
        f_values = []
        best_x = None
        best_f = np.inf
        calls = 0

        for p in points:
            if calls >= budget or time.time() > deadline - 10:
                break
            try:
                f_val = func(p)
            except:
                continue
            calls += 1
            f_values.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()

        if best_x is None:
            # Fallback if no evaluation succeeded (unlikely)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # 2. Local Optimization using Nelder-Mead
        # We will run multiple Nelder-Mead instances from the top K initial points
        # to avoid getting stuck in local minima or missing the ridge.
        
        # Sort initial points by function value
        valid_points = [(p, f) for p, f in zip(points, f_values) if not np.isnan(f)]
        valid_points.sort(key=lambda x: x[1])
        
        # Take top 3 points to start local searches
        top_k = min(3, len(valid_points))
        
        for i in range(top_k):
            if calls >= budget or time.time() > deadline - 10:
                break
            
            x0 = valid_points[i][0]
            f0 = valid_points[i][1]
            
            # Define a wrapper for func to track calls and enforce deadline
            def wrapped_func(x):
                nonlocal calls, best_f, best_x
                if calls >= budget or time.time() > deadline - 5:
                    raise Exception("Budget or time limit exceeded")
                x = np.asarray(x)
                if x.shape != (dim,):
                    x = x.reshape(dim)
                # Clip to box to be safe, although func does it
                x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            # Run Nelder-Mead
            # maxiter is set high, but we rely on the wrapper to stop it
            # xatol and fatol are tight to ensure convergence
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            except Exception:
                pass

            # If we still have budget, try a finer local search (Powell or Coordinate Descent)
            # Powell can be good for ridge functions if NM gets close.
            if calls < budget - 10 and time.time() < deadline - 10:
                try:
                    res2 = minimize(wrapped_func, best_x, method='Powell',
                                    options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-10})
                except Exception:
                    pass

        return best_x, best_f
