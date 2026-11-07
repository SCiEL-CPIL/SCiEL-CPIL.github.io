"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Optimizer combining random sampling, L-BFGS-B local optimization, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        # Track calls to ensure we don't exceed budget
        call_count = [0]
        best_x = [None]
        best_f = [float('inf')]

        def wrapped_func(x):
            if call_count[0] >= budget:
                return best_f[0]
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            call_count[0] += 1
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Random sampling to find a good starting point
        # Bent Cigar has a narrow valley; random sampling helps locate it
        n_samples = 15
        for i in range(n_samples):
            if time.time() > deadline or call_count[0] >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            wrapped_func(x)

        if best_x[0] is None:
            x0 = np.zeros(dim)
            wrapped_func(x0)
            best_x[0] = x0.copy()

        # Phase 2: Local optimization with L-BFGS-B
        bounds = [(-5.0, 5.0) for _ in range(dim)]
        remaining_budget = budget - call_count[0]

        if remaining_budget > 10 and time.time() < deadline:
            try:
                minimize(
                    wrapped_func,
                    best_x[0],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'maxfev': remaining_budget, 'ftol': 1e-15, 'gtol': 1e-15}
                )
            except Exception:
                pass

        # Phase 3: Final high-precision polish pass with Nelder-Mead
        remaining_budget = budget - call_count[0]
        if remaining_budget > 5 and time.time() < deadline and best_x[0] is not None:
            try:
                # Create a small simplex around the current best for local polish
                x0 = best_x[0].copy()
                step = 0.05
                simplex = [x0.copy()]
                for i in range(dim):
                    xi = x0.copy()
                    xi[i] += step
                    simplex.append(xi)
                
                minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining_budget, 200), 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        # Phase 4: If budget remains, do another L-BFGS-B polish
        remaining_budget = budget - call_count[0]
        if remaining_budget > 10 and time.time() < deadline and best_x[0] is not None:
            try:
                minimize(
                    wrapped_func,
                    best_x[0],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'maxfev': remaining_budget, 'ftol': 1e-15, 'gtol': 1e-15}
                )
            except Exception:
                pass

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
