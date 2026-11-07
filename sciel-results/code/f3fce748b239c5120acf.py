"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: few L-BFGS-B starts to find basin, then (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 45s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Quick L-BFGS-B from 3 diverse starts (limited budget each)
        # Rosenbrock has a narrow valley; gradient-based methods find it fast if started nearby
        n_lbfqsb_starts = 3
        starts = [
            np.zeros(self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
        ]

        for x0 in starts:
            if not time_ok() or call_count >= self.budget - 100:
                break
            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxls': 30, 'maxfun': 200}
                )
            except Exception:
                pass

        # Phase 2: (1+1)-ES with 1/5 success rule for the remaining budget
        # This is robust and efficient for the Rosenbrock valley geometry
        if time_ok() and call_count < self.budget - 5:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.5
            stuck = 0
            remaining = self.budget - call_count - 5

            for _ in range(remaining):
                if not time_ok() or call_count >= self.budget - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = tracked_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    stuck = 0
                else:
                    sigma *= 0.87
                    stuck += 1

                # Escape mechanism: if stuck too long, restart from best with perturbation
                if stuck > 40:
                    x = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
                    fx = tracked_func(x)
                    sigma = 0.3
                    stuck = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
