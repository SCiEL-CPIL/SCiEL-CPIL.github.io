"""Rewritten optimizer: Halton global sampling + Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def halton_sequence(n, dim, skip=10):
    """Generate n Halton sequence points in [0, 1)^dim, skipping first 'skip' for better spread."""
    points = []
    for i in range(skip, skip + n):
        point = []
        for base in range(2, dim + 2):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                f /= base
                r += f * (i_val % base)
                i_val //= base
            point.append(r)
        points.append(point)
    return np.array(points)


class Optimizer:
    """Halton global sampling for basin discovery + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Phase 1: Global sampling with Halton sequence
        # Use first 100 evaluations for global exploration
        n_global = min(100, budget // 5)
        halton_pts = halton_sequence(n_global, dim, skip=10)
        # Map from [0,1)^d to [-5,5]^d
        global_pts = halton_pts * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        evaluations = 0
        candidates = []

        for pt in global_pts:
            if evaluations >= budget:
                break
            if time.time() > deadline - 60:  # leave time for local refinement
                break
            x = np.clip(pt, -5, 5)
            fx = func(x)
            evaluations += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local refinement with Nelder-Mead from top candidates
        # Use remaining budget for local optimization
        remaining = budget - evaluations
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # Select top K candidates for local refinement
        # Distribute remaining budget among them
        n_local = min(5, len(candidates))
        budget_per_local = max(1, remaining // n_local)

        for idx in range(n_local):
            if evaluations >= budget:
                break
            if time.time() > deadline - 10:
                break

            start_x, start_f = candidates[idx]

            # If this is the best point, use more budget
            if idx == 0:
                local_budget = min(budget - evaluations, max(budget_per_local * 2, 50))
            else:
                local_budget = min(budget - evaluations, budget_per_local)

            if local_budget < 5:
                break

            # Nelder-Mead with step size control
            def objective(x):
                nonlocal evaluations
                x_clipped = np.clip(x, -5, 5)
                evaluations += 1
                return func(x_clipped)

            # Initial simplex: small perturbations around start point
            step = 0.5
            simplex = [start_x.copy()]
            for d in range(dim):
                perturbed = start_x.copy()
                perturbed[d] += step
                simplex.append(np.clip(perturbed, -5, 5))

            try:
                res = minimize(
                    objective,
                    start_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': local_budget,
                        'xatol': 1e-4,
                        'fatol': 1e-6,
                        'initial_simplex': simplex,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

            # If we've used all budget or hit deadline, stop
            if evaluations >= budget or time.time() > deadline - 10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
