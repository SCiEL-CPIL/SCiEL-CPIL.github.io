"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive multi-start local search with global exploration for f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(123)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def elapsed():
            return time.time() - start_time

        def time_ok():
            return elapsed() < deadline - 5

        def safe_eval(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or not time_ok():
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx if np.isfinite(fx) else float('inf')

        # Phase 1: Broad random sampling to identify promising basins
        n_random = min(50, budget // 10)
        candidates = []
        for i in range(n_random):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_eval(x)
            if np.isfinite(fx):
                candidates.append((fx, x.copy()))

        if best_x is None:
            best_x = np.zeros(dim)
            safe_eval(best_x)
            candidates.append((best_f, best_x.copy()))

        # Sort candidates by fitness (best first)
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Adaptive multi-start local search
        # Use Nelder-Mead from the best random points, with adaptive step sizes
        # and early stopping for local minima that don't improve.
        
        # We'll try local search from the top K candidates
        # K adapts based on budget remaining
        top_k = min(8, len(candidates), max(2, (budget - calls_made) // 60))
        
        for idx in range(top_k):
            if calls_made >= budget or not time_ok():
                break
            
            fx_start, x_start = candidates[idx]
            
            # Try Nelder-Mead with adaptive parameters
            # Use a smaller initial simplex for later candidates (more refined)
            initial_step = 1.5 if idx == 0 else 0.8 if idx < 3 else 0.4
            
            try:
                res = minimize(
                    safe_eval,
                    x_start.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 80,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True,
                        'initial_simplex': np.array([x_start + initial_step * rng.standard_normal(dim) if i == 0 else x_start + 0.1 * initial_step * rng.standard_normal(dim) for i in range(dim + 1)])
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            # After each local search, add the result to candidates if it improved
            if np.isfinite(res.fun) and res.fun < fx_start:
                candidates.append((res.fun, res.x.copy()))
                candidates.sort(key=lambda t: t[0])
        
        # Phase 3: Final fine-grained local refinement from the best point
        if calls_made < budget - 20 and time_ok() and best_x is not None:
            try:
                res = minimize(
                    safe_eval,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
