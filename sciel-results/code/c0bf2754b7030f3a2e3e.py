"""Black-box continuous optimizer on BBOB f12 (Bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Optimizer combining random initialization, L-BFGS-B, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track calls to ensure we don't exceed budget
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling to find a good starting point
        # Bent Cigar has a narrow valley; random sampling helps locate it
        rng = np.random.default_rng(42)
        n_samples = 30
        for i in range(n_samples):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped_func(x)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # Phase 2: Local optimization with L-BFGS-B
        bounds = [(-5.0, 5.0) for _ in range(dim)]
        remaining_budget = budget - call_count

        try:
            if remaining_budget > 10 and time.time() < deadline:
                result = minimize(
                    wrapped_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'maxfev': remaining_budget, 'ftol': 1e-15, 'gtol': 1e-15}
                )
        except Exception:
            pass

        # Phase 3: Final high-precision polish pass with Nelder-Mead
        remaining_budget = budget - call_count
        if remaining_budget > 5 and time.time() < deadline:
            try:
                x0 = best_x.copy()
                result2 = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining_budget, 100), 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        # Phase 4: Very fine local search with tiny steps to squeeze out last bits
        remaining_budget = budget - call_count
        if remaining_budget > 0 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.001

            for _ in range(remaining_budget):
                if call_count >= budget or time.time() > deadline:
                    break
                trial = np.clip(x + sigma_local * rng.standard_normal(dim), -5.0, 5.0)
                f_trial = wrapped_func(trial)
                if f_trial < fx:
                    x = trial.copy()
                    fx = f_trial
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9
                sigma_local = max(sigma_local, 1e-10)

        return best_x, best_f

# EVOLVE-BLOCK-END
