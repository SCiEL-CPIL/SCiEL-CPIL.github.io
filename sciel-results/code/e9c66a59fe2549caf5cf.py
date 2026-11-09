"""Optimized solver for BBOB f9 (rotated Rosenbrock) using multi-start L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer with adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Wrapper that clips to box and counts calls
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

        # Multi-start strategy: generate diverse starting points
        rng = np.random.default_rng(42)

        # Start points: prioritize origin-near points for Rosenbrock
        starts = []

        # 1. Origin and near-origin points (highest priority for Rosenbrock)
        starts.append(np.zeros(dim))
        for _ in range(15):
            starts.append(rng.normal(0, 1.0, dim))
        for scale in [0.5, 1.0, 2.0, 3.0]:
            starts.append(np.full(dim, scale))
            starts.append(np.full(dim, -scale))

        # 2. Points along axes (to handle rotations)
        for i in range(dim):
            for val in [-3, -2, -1, 1, 2, 3]:
                p = np.zeros(dim)
                p[i] = val
                starts.append(p)

        # 3. Structured points for rotated problems
        for s in [-3, -2, -1, 0, 1, 2, 3]:
            starts.append(np.full(dim, s))
        
        # 4. Random points in likely minimum region
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, dim))

        # 5. Broader random exploration
        for _ in range(8):
            starts.append(rng.uniform(-5, 5, dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.allclose(s, u, atol=1e-6) for u in unique_starts):
                unique_starts.append(s)

        # Sort: prefer points closer to origin (Rosenbrock min is at origin pre-rotation)
        starts = sorted(unique_starts, key=lambda s: np.sum(s**2))

        bounds = [(-5.0, 5.0)] * dim

        # Track best solution
        overall_best_x = None
        overall_best_f = np.inf
        converged_points = []

        # Phase 1: Multi-start with adaptive iteration limits
        for idx, start in enumerate(starts):
            if time.time() > deadline - 15.0:
                break

            if call_count[0] >= budget:
                break

            # Adaptive maxiter: more iterations for early starts, fewer for later
            remaining_budget = budget - call_count[0]
            if remaining_budget < 20:
                break

            # Allocate iterations based on position and remaining budget
            if idx < 5:
                maxiter = min(150, max(30, remaining_budget // 3))
            elif idx < 15:
                maxiter = min(100, max(20, remaining_budget // 4))
            else:
                maxiter = min(50, max(10, remaining_budget // 5))

            # Tighter tolerance for later refinements
            if overall_best_f < np.inf and overall_best_f < 1.0:
                ftol = 1e-14
                gtol = 1e-12
            else:
                ftol = 1e-12
                gtol = 1e-10

            try:
                res = minimize(
                    counted_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
                )

                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
                    converged_points.append(overall_best_x.copy())

            except Exception:
                continue

        # Phase 2: High-precision polishing on best solution with multiple passes
        if overall_best_x is not None and call_count[0] < budget and time.time() < deadline - 10.0:
            # Multiple refinement passes with increasing precision
            for polish_pass in range(5):
                if call_count[0] >= budget or time.time() > deadline - 10.0:
                    break
                
                remaining = budget - call_count[0]
                if remaining < 10:
                    break
                
                # Tighter tolerances each pass
                ftol = 1e-15
                gtol = 1e-13
                maxiter = min(100, max(10, remaining // 2))

                try:
                    res = minimize(
                        counted_func,
                        overall_best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
                    )
                    if res.fun < overall_best_f:
                        overall_best_f = res.fun
                        overall_best_x = np.clip(res.x, -5, 5)
                    else:
                        # If no improvement, try a smaller perturbation
                        if remaining > 20:
                            perturb = rng.normal(0, 1e-4, dim)
                            perturbed_x = np.clip(overall_best_x + perturb, -5, 5)
                            res2 = minimize(
                                counted_func,
                                perturbed_x,
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': min(50, max(10, remaining // 3)), 'ftol': ftol, 'gtol': gtol}
                            )
                            if res2.fun < overall_best_f:
                                overall_best_f = res2.fun
                                overall_best_x = np.clip(res2.x, -5, 5)
                except Exception:
                    break

        # Phase 3: If time/budget remains, try local search from converged points
        if overall_best_x is not None and call_count[0] < budget and time.time() < deadline - 5.0:
            for cp in converged_points[-5:]:  # Last 5 converged points
                if call_count[0] >= budget or time.time() > deadline - 5.0:
                    break
                if np.allclose(cp, overall_best_x, atol=1e-4):
                    continue
                remaining = budget - call_count[0]
                if remaining < 15:
                    break
                try:
                    res = minimize(
                        counted_func,
                        cp,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': min(50, max(10, remaining // 2)), 'ftol': 1e-15, 'gtol': 1e-13}
                    )
                    if res.fun < overall_best_f:
                        overall_best_f = res.fun
                        overall_best_x = np.clip(res.x, -5, 5)
                except Exception:
                    continue

        # Return the best solution found
        if overall_best_x is None:
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f
