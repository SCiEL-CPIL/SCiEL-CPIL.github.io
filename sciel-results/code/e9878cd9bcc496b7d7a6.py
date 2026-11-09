"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Combined LHS + Powell + Nelder-Mead optimizer for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        n = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = float('inf')
        all_evals = []  # (x, f) pairs for potential multi-restart

        def wrapped_func(x):
            nonlocal best_x, best_f
            if calls[0] >= budget or time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            all_evals.append((x_clipped.copy(), val))
            return val

        # Phase 1: Latin Hypercube Sampling for broad exploration
        n_lhs = 30
        if budget > n_lhs:
            rng = np.random.default_rng(12345)
            lhs_points = np.zeros((n_lhs, n))
            for d in range(n):
                # For each dimension, create n_lhs bins and pick one random point per bin
                perm = rng.permutation(n_lhs)
                for i in range(n_lhs):
                    low = -5 + (10 * perm[i] / n_lhs)
                    high = -5 + (10 * (perm[i] + 1) / n_lhs)
                    lhs_points[i, d] = rng.uniform(low, high)

            # Evaluate all LHS points
            for x in lhs_points:
                if calls[0] >= budget or time.time() > deadline:
                    break
                wrapped_func(x)

        if best_x is None:
            # Fallback: evaluate origin
            x0 = np.zeros(n)
            best_f = func(x0)
            best_x = x0.copy()
            calls[0] = 1
            all_evals.append((x0.copy(), best_f))

        if calls[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Powell's Method for local refinement from best point
        remaining = budget - calls[0]
        if remaining > 20:
            try:
                res = minimize(wrapped_func, best_x, method='Powell',
                               bounds=[(-5, 5)] * n,
                               options={'maxiter': min(500, remaining),
                                        'xatol': 1e-10, 'fatol': 1e-14})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        if calls[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: If budget remains, try second Powell from 2nd-best LHS point
        remaining = budget - calls[0]
        if remaining > 30 and len(all_evals) > 1:
            # Sort all evaluations by function value
            sorted_evals = sorted(all_evals, key=lambda t: t[1])
            # Find the 2nd best distinct point
            second_best_x = None
            for x, f in sorted_evals:
                if np.linalg.norm(x - best_x) > 0.1:
                    second_best_x = x
                    break
            if second_best_x is not None:
                try:
                    res2 = minimize(wrapped_func, second_best_x, method='Powell',
                                    bounds=[(-5, 5)] * n,
                                    options={'maxiter': min(200, remaining // 2),
                                             'xatol': 1e-10, 'fatol': 1e-14})
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = np.clip(res2.x, -5, 5).copy()
                except Exception:
                    pass

        if calls[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Fine-tuning with Nelder-Mead for final precision
        remaining = budget - calls[0]
        if remaining > 10:
            try:
                # Scale the initial simplex based on problem scale
                initial_simplex = 0.05
                res_nm = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                  options={'maxiter': min(300, remaining),
                                           'xatol': 1e-12, 'fatol': 1e-16,
                                           'adaptive': True,
                                           'initial_simplex': initial_simplex})
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = np.clip(res_nm.x, -5, 5).copy()
            except Exception:
                pass

        if calls[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 5: Ultra-fine coordinate descent polish if budget remains
        remaining = budget - calls[0]
        if remaining > 5:
            step = 0.01
            for _ in range(min(remaining, 50)):
                if time.time() > deadline or calls[0] >= budget:
                    break
                improved = False
                for d in range(n):
                    for direction in [-step, step]:
                        x_try = best_x.copy()
                        x_try[d] += direction
                        if np.all(x_try >= -5) and np.all(x_try <= 5):
                            val = wrapped_func(x_try)
                            if val < best_f - 1e-15:
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break

        # Ensure best_x is within bounds
        best_x = np.clip(best_x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
