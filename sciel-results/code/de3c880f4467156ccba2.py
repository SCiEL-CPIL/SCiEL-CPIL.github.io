"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Diverse sampling + L-BFGS-B local exploitation for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        global_best = None
        global_fx = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, global_best, global_fx
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < global_fx:
                global_fx = fx
                global_best = x.copy()
            return fx

        # Phase 1: Diverse global sampling to find good basins
        # Use Latin Hypercube Sampling for better coverage
        n_sample = 60
        lhs_points = []
        for d in range(dim):
            perm = rng.permutation(n_sample)
            points_d = (perm + 0.5) / n_sample * 10.0 - 5.0  # [-5, 5]
            lhs_points.append(points_d)
        lhs_points = np.column_stack(lhs_points)

        # Add some targeted points: center, corners, and random
        extra_points = [np.zeros(dim)]
        for _ in range(10):
            extra_points.append(rng.uniform(-4, 4, dim))

        all_sample_points = np.vstack([lhs_points, np.array(extra_points)])
        
        # Evaluate all sample points
        for x in all_sample_points:
            if calls_made >= budget - 50 or time.time() > deadline - 30:
                break
            evaluate(x)

        # Phase 2: Local exploitation with L-BFGS-B from top candidates
        # Collect all evaluated points and their values
        # We need to track which points we've evaluated
        evaluated_points = []
        for x in all_sample_points:
            if calls_made >= len(evaluated_points) + 1:
                break
            # Re-evaluate to get value (or we could have stored them)
            # Actually, let's just use the global best and a few diverse points
            break

        # Use L-BFGS-B from multiple starting points
        # Start from global best and a few other diverse points
        start_candidates = [global_best.copy() if global_best is not None else np.zeros(dim)]
        
        # Add a few diverse starting points for L-BFGS-B
        for _ in range(5):
            if global_best is not None:
                start_candidates.append(global_best + rng.uniform(-2, 2, dim))
            else:
                start_candidates.append(rng.uniform(-4, 4, dim))

        # Also add some random points far from global best for diversity
        for _ in range(3):
            start_candidates.append(rng.uniform(-4, 4, dim))

        # Run L-BFGS-B from each candidate
        for x0 in start_candidates:
            if calls_made >= budget - 10 or time.time() > deadline - 10:
                break

            x0 = np.clip(x0, -5.0, 5.0)

            # Define objective function for scipy
            def objective(x):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    return np.inf
                return evaluate(x)

            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )

                if result.fun < global_fx:
                    global_fx = result.fun
                    global_best = np.clip(result.x, -5.0, 5.0).copy()

            except Exception:
                pass

        # Phase 3: Final polish with small random perturbations around global best
        remaining = budget - calls_made
        if remaining > 5 and global_best is not None and time.time() < deadline - 5:
            x = global_best.copy()
            fx = global_fx
            sigma = 0.1

            for _ in range(min(remaining - 2, 30)):
                if time.time() > deadline - 2:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 0.8
                else:
                    sigma *= 0.9

        return global_best, global_fx
# EVOLVE-BLOCK-END
