"""Direct solver for BBOB f9 (rotated Rosenbrock) exploiting valley structure."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with Rosenbrock-aware start points and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        rng = np.random.default_rng(123)

        # Phase 1: Targeted exploration
        # Instead of a large random/grid search, use a small set of strategic points.
        # The origin is a good candidate for BBOB functions.
        # Points along the main diagonal and axes can help if the rotation is simple.
        strategic_starts = [np.zeros(dim)]
        for s in [-2, 2]:
            strategic_starts.append(np.full(dim, s))
            for i in range(dim):
                p = np.zeros(dim)
                p[i] = s
                strategic_starts.append(p)
        
        # Add a few random points to cover the box more broadly
        for _ in range(5):
            strategic_starts.append(rng.uniform(-4, 4, dim))

        # Evaluate these strategic starts
        for x0 in strategic_starts:
            if call_count[0] >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            counted_func(np.clip(x0, -5, 5))

        # Ensure we have a best point
        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            best_f[0] = counted_func(best_x[0])

        # Phase 2: Local optimization with L-BFGS-B
        # Start from the best point found so far, and a few other good points.
        # We will re-evaluate the best point to ensure it's the current best.
        
        # Prepare a list of starts for local optimization.
        # 1. The best point found.
        # 2. The origin (often a good start for BBOB).
        # 3. A few points near the best point.
        
        starts_for_local = [best_x[0].copy()]
        starts_for_local.append(np.zeros(dim))
        
        # Add perturbations around the best point
        for _ in range(3):
            perturbed = best_x[0] + rng.normal(0, 0.5, dim)
            starts_for_local.append(np.clip(perturbed, -5, 5))

        # Run L-BFGS-B from each start
        for x0 in starts_for_local:
            if call_count[0] >= budget - 50:  # Leave budget for final polish
                break
            if time.time() > deadline - 20.0:
                break

            try:
                res = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5, 5)
            except Exception:
                continue

        # Phase 3: Final high-precision polish
        # If we have budget left, run a very tight L-BFGS-B from the best point.
        if best_x[0] is not None and call_count[0] < budget - 10 and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    counted_func,
                    best_x[0],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Ensure we have a valid result
        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            if call_count[0] < budget:
                best_f[0] = counted_func(best_x[0])
            else:
                best_f[0] = 0.0 # Fallback, should not happen

        return best_x[0], best_f[0]
