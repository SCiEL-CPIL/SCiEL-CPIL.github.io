import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Generate diverse starts
        # 1. The standard Schwefel minimum
        schwefel_min = np.full(dim, 4.20968746)
        starts = [schwefel_min]
        
        # 2. Random points
        # 3. Points near boundaries
        # 4. Zero point
        starts.append(np.zeros(dim))
        
        for _ in range(15):
            starts.append(rng.uniform(-5, 5, dim))
            
        # Evaluate all starts
        evals = []
        for x0 in starts:
            fx = try_func(x0)
            if fx is not None:
                evals.append((fx, x0))
        
        if not evals:
            return best_x, best_f
            
        # Sort by fitness
        evals.sort(key=lambda t: t[0])
        
        # Take top 3 distinct-ish points
        top_points = []
        for f_val, x_val in evals:
            if all(np.linalg.norm(x_val - s) > 1.0 for s in [p[1] for p in top_points]):
                top_points.append((f_val, x_val))
            if len(top_points) >= 3:
                break
        
        # Run Nelder-Mead from top points
        remaining_budget = budget - calls_made
        if remaining_budget > 10:
            budget_per_start = max(10, remaining_budget // len(top_points))
            
            for f0, x0 in top_points:
                if calls_made >= budget - 5:
                    break
                if time.time() > deadline:
                    break
                
                def scipy_func(x):
                    if calls_made >= budget:
                        return best_f
                    fx = try_func(x)
                    if fx is None:
                        return best_f
                    return fx
                
                try:
                    # Nelder-Mead options
                    # maxfev controls function evaluations
                    res = minimize(
                        scipy_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 1000,
                            'maxfev': budget_per_start,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                    if res.x is not None and res.fun is not None and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
