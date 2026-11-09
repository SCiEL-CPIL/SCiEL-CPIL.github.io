"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """LHS + adaptive (1+1)-ES with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Latin Hypercube Sampling for global exploration
        n_lhs = 80
        calls = 0
        best_x = None
        best_f = np.inf

        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for j in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, j] = (perm + 0.5) / n_lhs * 10.0 - 5.0  # uniform in [-5, 5]

        for pt in lhs_points:
            if calls >= budget or not time_ok():
                break
            f_val = func(pt)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # Phase 2: Adaptive (1+1)-ES
        x = best_x.copy()
        f_current = best_f
        sigma = 0.5
        success_count = 0
        failure_count = 0
        stuck_count = 0

        while calls < budget and time_ok():
            # Generate candidate
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)

            f_y = func(y)
            calls += 1

            if f_y < f_current:
                # Success
                x = y
                f_current = f_y
                if f_current < best_f:
                    best_f = f_current
                    best_x = x.copy()
                success_count += 1
                failure_count = 0
                stuck_count = 0
                # Increase sigma on success
                sigma *= 1.05
                if sigma > 5.0:
                    sigma = 5.0
            else:
                # Failure
                failure_count += 1
                stuck_count += 1
                success_count = 0
                # Decrease sigma on failure
                sigma *= 0.95
                if sigma < 0.001:
                    sigma = 0.001

            # If stuck for a while, try a larger jump
            if stuck_count > 30:
                # Random restart from best
                jump_sigma = 1.0
                z = rng.standard_normal(dim)
                y = best_x + jump_sigma * z
                y = np.clip(y, -5, 5)
                f_y = func(y)
                calls += 1
                if f_y < f_current:
                    x = y
                    f_current = f_y
                    if f_current < best_f:
                        best_f = f_current
                        best_x = x.copy()
                stuck_count = 0
                sigma = 0.5

        # Phase 3: Nelder-Mead polish on the best point found
        if calls < budget and time_ok():
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead for final polish
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
