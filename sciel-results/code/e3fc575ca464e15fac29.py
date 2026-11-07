"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: global random search + Nelder-Mead + L-BFGS-B polish for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        start_time = time.time()
        deadline = start_time + 285.0

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return 1e10
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        bounds = [(-5.0, 5.0)] * dim

        # Phase 1: Global random sampling to find promising regions
        # Use ~80 calls for broad exploration
        n_global = min(80, budget)
        for i in range(n_global):
            if time.time() > deadline or call_count >= budget:
                break
            if i < 5:
                # Structured starts near origin (likely close to shifted minimum)
                x0 = rng.normal(0, 1.0, dim)
            else:
                x0 = rng.uniform(-4.5, 4.5, dim)
            wrapped_func(x0)

        # Phase 2: Nelder-Mead from top candidates
        # NM uses dim+1=6 function evals per iteration, no gradient overhead
        # Run NM from several diverse starts
        nm_starts = [best_x.copy()]
        # Add a few more diverse starts
        for _ in range(3):
            nm_starts.append(rng.uniform(-3.0, 3.0, dim))

        for x0 in nm_starts:
            if time.time() > deadline or call_count >= budget:
                break
            remaining = budget - call_count
            if remaining < 30:
                break
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': min(200, remaining // 6), 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                continue

        # Phase 3: L-BFGS-B polish (most efficient near optimum, ~3 calls/iter)
        if call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            if remaining > 15:
                try:
                    res_final = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': remaining, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
                    )
                    if res_final.fun < best_f:
                        best_f = res_final.fun
                        best_x = res_final.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
