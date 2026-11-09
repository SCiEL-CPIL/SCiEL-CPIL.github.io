"""Rewritten optimizer: adaptive multi-start with Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f13 (sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        x_best = None
        fx_best = float('inf')
        calls_made = 0

        # Wrapper to count calls and track best
        def wrapper(x):
            nonlocal fx_best, x_best, calls_made
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()
            return fx

        # Phase 1: Coarse grid scan to find promising regions
        # Use a moderate grid to avoid burning too many calls
        n_grid = 3  # 3 points per dimension -> 3^5 = 243 evaluations
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    for l in range(n_grid):
                        for m in range(n_grid):
                            grid_points.append(np.array([
                                -5.0 + 10.0 * i / (n_grid - 1),
                                -5.0 + 10.0 * j / (n_grid - 1),
                                -5.0 + 10.0 * k / (n_grid - 1),
                                -5.0 + 10.0 * l / (n_grid - 1),
                                -5.0 + 10.0 * m / (n_grid - 1),
                            ]))

        # Evaluate grid points
        grid_results = []
        for gp in grid_points:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            fx = wrapper(gp)
            grid_results.append((fx, gp.copy()))

        if not grid_results:
            # Fallback: single random point
            if calls_made < self.budget and time.time() < deadline:
                x0 = np.random.default_rng(0).uniform(-5, 5, self.dim)
                wrapper(x0)

        # Sort grid results by fitness
        grid_results.sort(key=lambda t: t[0])

        # Phase 2: Local refinement from top grid points using Nelder-Mead
        # Select top K starting points for local optimization
        top_k = min(5, len(grid_results))
        for i in range(top_k):
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break

            x0, f0 = grid_results[i]
            # Use Nelder-Mead for local refinement
            # Limit iterations to conserve budget
            max_iter = max(10, (self.budget - calls_made) // (top_k - i))
            max_iter = min(max_iter, 200)

            try:
                result = minimize(
                    wrapper,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'maxfev': min(max_iter * self.dim, self.budget - calls_made),
                        'xatol': 1e-6,
                        'fatol': 1e-10,
                    }
                )
            except Exception:
                continue

            if calls_made >= self.budget:
                break

        # Phase 3: If budget remains, do a final coordinate-wise line search
        if x_best is not None and calls_made < self.budget and time.time() < deadline:
            # Simple coordinate descent with golden-section search
            x_curr = x_best.copy()
            f_curr = fx_best
            remaining = self.budget - calls_made

            for axis in range(self.dim):
                if remaining <= 5:
                    break
                if time.time() > deadline:
                    break

                # Golden-section search along this axis
                lo = -5.0
                hi = 5.0
                # Narrow initial range around current best
                span = 2.0
                lo = max(-5.0, x_curr[axis] - span)
                hi = min(5.0, x_curr[axis] + span)

                invphi = (np.sqrt(5.0) - 1.0) / 2.0
                invphi2 = invphi * invphi

                x1 = hi - invphi * (hi - lo)
                x2 = lo + invphi * (hi - lo)

                test_x1 = x_curr.copy()
                test_x1[axis] = x1
                f1 = wrapper(test_x1)
                remaining -= 1

                test_x2 = x_curr.copy()
                test_x2[axis] = x2
                f2 = wrapper(test_x2)
                remaining -= 1

                # Iterate golden-section
                for _ in range(min(30, remaining - 1)):
                    if time.time() > deadline:
                        break
                    if f1 < f2:
                        hi = x2
                        x2 = x1
                        f2 = f1
                        x1 = hi - invphi * (hi - lo)
                        test_x = x_curr.copy()
                        test_x[axis] = x1
                        f1 = wrapper(test_x)
                        remaining -= 1
                    else:
                        lo = x1
                        x1 = x2
                        f1 = f2
                        x2 = lo + invphi * (hi - lo)
                        test_x = x_curr.copy()
                        test_x[axis] = x2
                        f2 = wrapper(test_x)
                        remaining -= 1

                # Update best if improved
                if f1 < f_curr:
                    x_curr[axis] = x1
                    f_curr = f1
                    x_best = x_curr.copy()
                    fx_best = f_curr
                elif f2 < f_curr:
                    x_curr[axis] = x2
                    f_curr = f2
                    x_best = x_curr.copy()
                    fx_best = f_curr

        return x_best if x_best is not None else np.zeros(self.dim), fx_best
# EVOLVE-BLOCK-END
