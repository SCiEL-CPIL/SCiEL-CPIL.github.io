import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim

        # Phase 1: Global Search - evaluate strategic points to find basin
        # Origin + random points in [-4,4]^5 where minimizer is known to lie
        starts = [np.zeros(self.dim)]
        for _ in range(9):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))
        
        # Evaluate starts, leaving budget for local optimization
        for x0 in starts:
            if call_count >= self.budget - 100:
                break
            tracked_func(x0)
        
        # Phase 2: L-BFGS-B local optimization from best point
        # L-BFGS-B is highly effective for smooth Rosenbrock once in basin
        if time_ok() and call_count < self.budget - 5:
            try:
                x_start = best_x.copy()
                maxfun_remaining = min(self.budget - call_count - 5, 200)
                if maxfun_remaining < 10:
                    maxfun_remaining = 10
                    
                res = minimize(
                    tracked_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'maxfun': maxfun_remaining,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
            except Exception:
                pass
        
        # Phase 3: If budget remains, try a second L-BFGS-B from a different start
        # or Nelder-Mead polish for final refinement
        if time_ok() and call_count < self.budget - 10:
            try:
                remaining = self.budget - call_count
                if remaining > 20:
                    # Try Nelder-Mead for derivative-free polish
                    res = minimize(
                        tracked_func,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': remaining,
                            'adaptive': True,
                            'xatol': 1e-12,
                            'fatol': 1e-14
                        }
                    )
            except Exception:
                pass

        return best_x, best_fx
