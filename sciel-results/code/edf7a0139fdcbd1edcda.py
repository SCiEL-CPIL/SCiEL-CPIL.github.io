"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(n)
        best_f = np.inf
        call_count = 0

        def time_ok():
            return time.time() < deadline

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            call_count += 1
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluations at diverse points
        rng = np.random.default_rng(42)

        # Start points: origin, random, and a few structured points
        start_points = [np.zeros(n)]
        # Add random points
        for _ in range(4):
            start_points.append(rng.uniform(-4, 4, n))
        # Add axis-aligned points
        for i in range(n):
            pt = np.zeros(n)
            pt[i] = rng.uniform(-4, 4)
            start_points.append(pt)

        # Evaluate initial points
        for sp in start_points:
            if call_count >= budget or not time_ok():
                break
            f = safe_call(sp)

        if call_count >= budget or not time_ok():
            return best_x, best_f

        # Multi-start Nelder-Mead
        remaining_budget = budget - call_count
        n_starts = min(5, max(1, remaining_budget // 40))

        for s in range(n_starts):
            if call_count >= budget or not time_ok():
                break

            # Choose starting point: best so far or a new random point
            if s == 0:
                x0 = best_x.copy()
            else:
                x0 = rng.uniform(-4, 4, n)

            # Set up Nelder-Mead options
            # Limit function evaluations per restart
            maxfev = min(remaining_budget - (budget - call_count), max(10, (budget - call_count) // max(1, n_starts - s)))

            # Small initial simplex for exploration
            x0_clipped = np.clip(x0, -4.9, 4.9)

            # Use a wrapper that counts calls and respects deadline
            def nelder_func(x):
                if not time_ok() or call_count >= budget:
                    return 1e10
                return safe_call(x)

            try:
                res = minimize(
                    nelder_func,
                    x0_clipped,
                    method='Nelder-Mead',
                    options={
                        'maxfev': maxfev,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

            if call_count >= budget or not time_ok():
                break

            # Polish with a tighter Nelder-Mead around the best found
            if time_ok() and call_count < budget:
                remaining = budget - call_count
                if remaining > 5:
                    polish_maxfev = min(remaining, 20)
                    x0_polish = np.clip(best_x, -4.9, 4.9)

                    def polish_func(x):
                        if not time_ok() or call_count >= budget:
                            return 1e10
                        return safe_call(x)

                    try:
                        res2 = minimize(
                            polish_func,
                            x0_polish,
                            method='Nelder-Mead',
                            options={
                                'maxfev': polish_maxfev,
                                'xatol': 1e-10,
                                'fatol': 1e-12,
                                'adaptive': True
                            }
                        )
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
