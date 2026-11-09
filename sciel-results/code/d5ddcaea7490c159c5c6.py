"""Hybrid Nelder-Mead with adaptive restarts and final gradient polish for BBOB f12."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead optimizer with adaptive restarts for smooth black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin vs. 330 s kill
        max_calls = budget

        calls = [0]
        best_fx = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def f(x):
            nonlocal best_fx, best_x
            if calls[0] >= max_calls:
                return float('inf')
            calls[0] += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                val = float(func(x_clipped))
            except Exception:
                return float('inf')
            if not np.isfinite(val):
                return float('inf')
            if val < best_fx:
                best_fx = val
                best_x = x_clipped.copy()
            return val

        def remaining():
            return deadline - time.time()

        def run_powell(x0):
            """Run Powell's method from x0."""
            if calls[0] >= max_calls or remaining() < 5.0:
                return x0, f(x0)
            
            # Powell method options
            # maxiter is not strictly call count, but we monitor calls in f
            # maxfun is not directly supported in all versions or might not map 1:1 to calls due to internal logic
            # We rely on the global call counter and time check.
            res = minimize(f, x0, method='Powell', options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8})
            if res.success or np.isfinite(res.fun):
                return res.x, res.fun
            return x0, f(x0)

        # Phase 1: Multi-start Powell
        # Start at origin
        x0 = np.zeros(dim)
        fx0 = f(x0)
        x_best, fx_best = run_powell(x0)
        
        # If we have budget for more starts, try random points
        # Powell is robust, so 1-2 starts are often enough, but random starts help if origin is far.
        if calls[0] < max_calls * 0.5 and remaining() > 50.0:
            x1 = rng.uniform(-4, 4, dim)
            fx1 = f(x1)
            x1_res, fx1_res = run_powell(x1)
            if fx1_res < fx_best:
                x_best, fx_best = x1_res, fx1_res

        if calls[0] < max_calls * 0.8 and remaining() > 30.0:
            x2 = rng.uniform(-4, 4, dim)
            fx2 = f(x2)
            x2_res, fx2_res = run_powell(x2)
            if fx2_res < fx_best:
                x_best, fx_best = x2_res, fx2_res

        # Phase 2: Final Polish
        # Use a tight Nelder-Mead or simple coordinate descent to refine
        if calls[0] < max_calls - 10 and remaining() > 10.0:
            x_polish = x_best.copy()
            fx_polish = f(x_polish)
            
            # Small step Nelder-Mead for final convergence
            n = dim + 1
            simplex = [x_polish.copy()]
            f_values = [fx_polish]
            step = 1e-4 # Very small step for polish
            
            for i in range(dim):
                xi = x_polish.copy()
                xi[i] += step
                fi = f(xi)
                simplex.append(xi)
                f_values.append(fi)
                
            if calls[0] < max_calls:
                 # Run a few iterations of NM manually or use scipy NM with tight tol
                 res_polish = minimize(f, x_polish, method='Nelder-Mead', options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12, 'initial_simplex': simplex})
                 if np.isfinite(res_polish.fun) and res_polish.fun < fx_polish:
                     x_best, fx_best = res_polish.x, res_polish.fun

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(dim)
            if calls[0] < max_calls:
                best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
