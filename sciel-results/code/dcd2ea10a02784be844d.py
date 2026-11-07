"""Rewritten optimizer for BBOB f14 (Different powers) with multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer with deadline guard for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # State
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and call_count < budget

        # Phase 1: Initial diverse evaluations to get a good starting point
        n_init = min(20, budget // 5)
        # Use a mix of strategies for initial points
        init_points = []

        # Center point
        init_points.append(np.zeros(dim))

        # Random points in the box
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-5.0, 5.0, size=dim))

        for i in range(len(init_points)):
            if not time_ok():
                break
            evaluate(init_points[i])

        if not time_ok() or best_x is None:
            if best_x is None:
                best_x = np.zeros(dim)
                evaluate(best_x)
            return best_x, best_f

        # Phase 2: Multi-start L-BFGS-B optimization
        # Use the best found point and several other promising points as starting points
        # Sort initial points by their fitness
        init_fitness = []
        for p in init_points:
            p_clipped = np.clip(p, -5.0, 5.0)
            init_fitness.append((p_clipped, evaluate(p_clipped)))

        # Sort by fitness
        init_fitness.sort(key=lambda t: t[1])

        # Get top candidates for L-BFGS-B starts
        n_starts = min(15, len(init_fitness))
        starts = [init_fitness[i][0] for i in range(n_starts)]

        # Also add some perturbed versions of the best point
        for _ in range(min(5, budget - call_count)):
            if not time_ok():
                break
            perturbed = best_x + rng.normal(0, 0.5, size=dim)
            perturbed = np.clip(perturbed, -5.0, 5.0)
            starts.append(perturbed)

        # Run L-BFGS-B from each starting point
        for start_point in starts:
            if not time_ok():
                break
            try:
                res = minimize(
                    lambda xx: evaluate(np.clip(xx, -5, 5)),
                    start_point,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: Fine-tuning with smaller perturbations around the best point
        if time_ok():
            for _ in range(min(10, budget - call_count)):
                if not time_ok():
                    break
                try:
                    perturbed = best_x + rng.normal(0, 0.1, size=dim)
                    perturbed = np.clip(perturbed, -5.0, 5.0)
                    res = minimize(
                        lambda xx: evaluate(np.clip(xx, -5, 5)),
                        perturbed,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Phase 4: Final Nelder-Mead polish if time permits
        if time_ok() and best_x is not None:
            try:
                res = minimize(
                    lambda xx: evaluate(np.clip(xx, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
