import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start hybrid optimizer: coarse grid + CMA-ES + L-BFGS-B for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        # Safety margin: 330s limit, stop at 320s
        deadline = time.time() + 320
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        bounds = [(-5, 5) for _ in range(dim)]
        
        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if x.shape != (dim,):
                x = x.reshape(-1)
            if len(x) != dim:
                x = x[:dim]
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def polish(x_start, max_iter=30, maxfev=None):
            """Run L-BFGS-B polish from a starting point."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            if maxfev is None:
                maxfev = max(10, (budget - calls) // 3)
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start.copy(), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': max_iter, 
                        'maxfev': maxfev, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8, 
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 1: Dense random sampling to get good initial candidates
        # Spend ~40% of budget on initial sampling
        n_samples = min(120, budget // 3)
        
        for i in range(n_samples):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            wrapped_func(x)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Run multiple L-BFGS-B from top diverse candidates
        # We'll do several rounds of local optimization
        
        # First, do a quick polish from the current best
        polish(best_x, max_iter=30, maxfev=max(20, (budget - calls) // 4))
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Try to find other basins using random restarts + polish
        n_restarts = 6
        for i in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Generate a random start, biased towards the current best with some spread
            x0 = best_x + rng.normal(0, 1.5, dim)
            x0 = np.clip(x0, -5, 5)
            
            # Evaluate start
            fx0 = wrapped_func(x0)
            
            if calls >= budget or time.time() > deadline:
                break
            
            # Polish from this start
            remaining = budget - calls
            if remaining < 15:
                break
            local_budget = max(15, min(remaining // 2, 50))
            polish(x0, max_iter=20, maxfev=local_budget)
            
            if calls >= budget or time.time() > deadline:
                break

        # Phase 4: If budget remains, do a final CMA-ES-like local search
        if calls < budget - 50 and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 50:
                # Do a small CMA-ES run
                try:
                    # Use a simple local search with shrinking step
                    sigma = 0.5
                    for _ in range(min(40, remaining // 2)):
                        if calls >= budget or time.time() > deadline:
                            break
                        x_try = best_x + rng.normal(0, sigma, dim)
                        x_try = np.clip(x_try, -5, 5)
                        fx_try = wrapped_func(x_try)
                        if fx_try < best_f:
                            best_f = fx_try
                            best_x = x_try.copy()
                        else:
                            sigma *= 0.95
                except Exception:
                    pass

        # Final polish if budget remains
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func, 
                        best_x.copy(), 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={
                            'maxiter': 50, 
                            'maxfev': remaining, 
                            'ftol': 1e-14, 
                            'gtol': 1e-10, 
                            'disp': False
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
