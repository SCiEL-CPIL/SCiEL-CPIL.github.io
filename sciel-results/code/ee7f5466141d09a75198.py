"""Multi-start Nelder-Mead with adaptive restarts for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget

        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Latin hypercube sampling for diverse initial points
        n_lhs = 48
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        # Evaluate LHS points
        for pt in lhs_points:
            if eval_count >= budget - 50:
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # Phase 2: Multi-start Nelder-Mead from best LHS points
        # Collect results from LHS
        lhs_results = []
        for pt in lhs_points:
            if eval_count >= budget - 50:
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            lhs_results.append((f_val, pt.copy()))

        # Sort by fitness
        lhs_results.sort(key=lambda x: x[0])

        # Start Nelder-Mead from top diverse points
        n_starts = min(12, len(lhs_results))
        start_points = []
        for i in range(n_starts):
            f_val, pt = lhs_results[i]
            # Check diversity
            is_diverse = True
            for sp in start_points:
                if np.linalg.norm(pt - sp) < 0.5:
                    is_diverse = False
                    break
            if is_diverse:
                start_points.append(pt)

        # Also add perturbed versions of the best point
        best_pt = lhs_results[0][1] if lhs_results else np.zeros(dim)
        for scale in [1.0, 0.5, 0.2]:
            sp = best_pt + rng.normal(0, scale, dim)
            sp = np.clip(sp, -5.0, 5.0)
            if sp not in start_points:
                start_points.append(sp)

        # Phase 3: Run Nelder-Mead from each starting point
        for sp in start_points:
            if eval_count >= budget - 20:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: Adaptive local random search with shrinking step size
        if eval_count < budget and time.time() < deadline:
            step_size = 0.3
            no_improve = 0
            max_no_improve = 20

            while eval_count < budget and time.time() < deadline:
                x = best_x + rng.normal(0, step_size, dim)
                x = np.clip(x, -5.0, 5.0)
                f_val = wrapped_func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        step_size *= 0.5
                        if step_size < 1e-7:
                            break
                        no_improve = 0

        # Phase 5: If budget remains, do more Nelder-Mead restarts from perturbed best
        remaining = budget - eval_count
        if remaining > 30 and time.time() < deadline:
            for _ in range(min(5, remaining // 10)):
                if eval_count >= budget - 10:
                    break
                if time.time() > deadline:
                    break
                sp = best_x + rng.normal(0, 0.3, dim)
                sp = np.clip(sp, -5.0, 5.0)
                try:
                    result = minimize(
                        wrapped_func,
                        sp,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
