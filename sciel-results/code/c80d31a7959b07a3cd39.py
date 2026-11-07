"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start BFGS with adaptive restarts for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        # Track best solution found
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Wrap func to count calls and enforce deadline
        def safe_func(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("deadline exceeded")
            x = np.clip(np.asarray(x, dtype=float).reshape(dim), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if not np.isfinite(fx):
                fx = 1e10
            return fx

        # Generate diverse starting points using a structured grid + random
        # For dim=5, we use a combination of:
        # 1. Origin and axis-aligned points
        # 2. Random points in [-4, 4]^dim
        # 3. Points near the typical Rosenbrock valley structure

        rng = np.random.default_rng(42)

        # Pre-generate a pool of candidate starting points
        start_points = []

        # Deterministic structured points
        start_points.append(np.zeros(dim))
        for i in range(dim):
            for val in [-3.0, -1.0, 1.0, 3.0]:
                p = np.zeros(dim)
                p[i] = val
                start_points.append(p)

        # Random points in [-4, 4]^dim
        n_random = 80
        random_pts = rng.uniform(-4.0, 4.0, size=(n_random, dim))
        start_points.extend([p for p in random_pts])

        # Points biased toward typical Rosenbrock valley (x_i close to 1 in unrotated)
        # But since rotation is hidden, just add more random coverage
        n_more_random = 120
        more_random = rng.uniform(-4.5, 4.5, size=(n_more_random, dim))
        start_points.extend([p for p in more_random])

        # Shuffle to get diversity early
        start_points = np.array(start_points)
        idx = rng.permutation(len(start_points))
        start_points = start_points[idx]

        # Multi-start BFGS
        # Budget allocation: each BFGS run uses up to ~15-30 function evaluations
        # We'll do multiple BFGS runs, each with a limited number of evaluations
        # Use options to limit max iterations and function calls per run

        max_evals_per_run = 25  # limit per BFGS run to allow many restarts
        max_iter_per_run = 50

        for start_idx in range(len(start_points)):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            x0 = start_points[start_idx].copy()
            x0 = np.clip(x0, -5.0, 5.0)

            try:
                # Use BFGS with limited evaluations
                res = minimize(
                    safe_func,
                    x0,
                    method='BFGS',
                    options={
                        'maxiter': max_iter_per_run,
                        'maxfun': max_evals_per_run,
                        'gtol': 1e-6,
                    }
                )

                if res.success or res.fun < best_f:
                    if np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()

            except (TimeoutError, Exception):
                if time.time() > deadline:
                    break
                # Continue to next start point if not a timeout

            # Check if we've used too many calls
            if calls_made >= budget:
                break

        # If we still have budget remaining, do a final local polish around best_x
        if best_x is not None and calls_made < budget and time.time() < deadline:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='BFGS',
                    options={
                        'maxiter': 100,
                        'maxfun': min(50, budget - calls_made),
                        'gtol': 1e-8,
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except (TimeoutError, Exception):
                pass

        # Final fallback: if no valid result, return a default
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x) if calls_made < budget else 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
