"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid Powell-Nelder-Mead optimizer for BBOB-style Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 30s safety margin before the 330s kill (or 300s limit if that's the hard stop)
        # The contract says "finish within 300 s", evaluator kills after 330s.
        # Let's aim for 290s to be safe.
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_fx = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_fx
            # Ensure x is a finite 1D array of size n
            if not np.all(np.isfinite(x)) or x.shape != (n,):
                return np.inf
            if calls >= budget:
                return np.inf
            try:
                val = func(x)
            except Exception:
                return np.inf
            calls += 1
            if np.isfinite(val) and val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        # Phase 1: Initial Exploration
        # We want to quickly identify a point close to the minimum.
        # Rosenbrock minimum in standard form is (1, ..., 1).
        # With shift/rotation, it's anywhere in [-4, 4]^5.
        # We sample a mix of structured and random points.
        
        initial_points = [
            np.zeros(n),
            np.ones(n),
            -np.ones(n),
            0.5 * np.ones(n),
            -0.5 * np.ones(n),
        ]
        
        # Add random points
        # Use a few more random points to ensure coverage
        for _ in range(15):
            initial_points.append(rng.uniform(-4, 4, n))
            
        for x in initial_points:
            if time.time() > deadline or calls >= budget:
                break
            f = safe_eval(x)
            # safe_eval updates best_x/best_fx if f is better
            
        if best_x is None:
            # Fallback
            best_x = np.zeros(n)
            safe_eval(best_x)

        # Phase 2: Powell Optimization
        # Powell is effective for Rosenbrock due to adaptive coordinate directions.
        remaining_budget = budget - calls
        if remaining_budget > 20 and time.time() < deadline - 5:
            
            # Wrapper for scipy that respects budget and time
            def powell_wrapper(x):
                nonlocal calls, best_x, best_fx
                if calls >= budget or time.time() > deadline:
                    raise ValueError("Budget or time limit exceeded")
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls += 1
                if np.isfinite(val) and val < best_fx:
                    best_fx = val
                    best_x = x_clipped.copy()
                return val

            try:
                res = scipy.optimize.minimize(
                    powell_wrapper,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 50, # Limit iterations to avoid getting stuck in slow convergence
                        'maxfev': remaining_budget,
                        'xtol': 1e-8,
                        'ftol': 1e-8,
                        'disp': False
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead Polish
        # Use NM for final high-precision polish.
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline - 2:
            
            def nm_wrapper(x):
                nonlocal calls, best_x, best_fx
                if calls >= budget or time.time() > deadline:
                    raise ValueError("Budget or time limit exceeded")
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls += 1
                if np.isfinite(val) and val < best_fx:
                    best_fx = val
                    best_x = x_clipped.copy()
                return val

            try:
                res = scipy.optimize.minimize(
                    nm_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'disp': False
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
