"""Optimizer for BBOB f23 (Katsuura) with LHS + (1+1)-ES + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """LHS initialization + (1+1)-ES with Nelder-Mead polish for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        call_count = 0

        # Phase 1: Latin Hypercube sampling to find good initial points
        n_lhs = min(30, budget // 10)
        lhs_points = self._latin_hypercube(rng, n_lhs, dim)

        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # initial step size

        successes = 0
        window = 10
        trial_count = 0

        remaining_budget = budget - call_count

        for i in range(remaining_budget):
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                sigma *= 1.15  # increase step size on success
            else:
                sigma *= 0.85  # decrease step size on failure

            # Keep track of global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma adjustment every window steps
            trial_count += 1
            if trial_count % window == 0:
                success_rate = successes / window
                if success_rate > 0.3:
                    sigma *= 1.3
                elif success_rate < 0.1:
                    sigma *= 0.7
                successes = 0

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)

        # Phase 3: Nelder-Mead polish on the best point found
        if time.time() < deadline - 5 and best_x is not None:
            try:
                # Use a smaller initial simplex for Nelder-Mead
                x0 = best_x.copy()
                # Nelder-Mead with tight tolerances
                result = minimize(
                    func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'maxiter': 500,
                        'maxfev': 200,
                        'initial_simplex': None  # default
                    }
                )
                if result.success or result.fun < best_f:
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f

    def _latin_hypercube(self, rng, n, dim):
        """Generate Latin Hypercube points in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            points[:, d] = -5 + 10 * (perm + rng.random(n)) / n
        return points
# EVOLVE-BLOCK-END
