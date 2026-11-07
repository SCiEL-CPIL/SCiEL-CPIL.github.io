"""Optimizer for BBOB-style Schwefel f20 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            fx = func(np.asarray(x, dtype=np.float64))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        # Phase 1: Multi-start CMA-ES
        # CMA-ES handles rotated landscapes well via covariance adaptation
        n_starts = 6
        start_points = []

        # Center
        start_points.append(np.zeros(self.dim))
        # Axis-aligned points
        for i in range(self.dim):
            p = np.zeros(self.dim)
            p[i] = 3.0
            start_points.append(p)
        # Additional random points
        while len(start_points) < n_starts:
            p = rng.uniform(-4, 4, self.dim)
            if all(np.linalg.norm(p - q) > 1.5 for q in start_points):
                start_points.append(p)

        # Allocate budget: ~60% to CMA-ES, ~40% to L-BFGS-B polish
        cmaes_budget = int(self.budget * 0.6)
        per_start_budget = max(20, cmaes_budget // n_starts)

        for sp in start_points:
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break

            cmaes_calls_this = [0]

            def cmaes_obj(x):
                nonlocal cmaes_calls_this
                if cmaes_calls_this[0] >= per_start_budget:
                    return best_f
                cmaes_calls_this[0] += 1
                return eval_func(x)

            try:
                res = minimize(
                    cmaes_obj,
                    sp,
                    method='Nelder-Mead',
                    options={
                        'maxiter': per_start_budget,
                        'xatol': 1e-4,
                        'fatol': 1e-6,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        # Phase 2: L-BFGS-B polish from best point
        if best_x is not None and call_count < self.budget:
            remaining = self.budget - call_count
            if remaining >= 10:
                bounds = [(-5.0, 5.0)] * self.dim
                lbfgs_calls = [0]

                def lbfgs_obj(x):
                    nonlocal lbfgs_calls
                    if lbfgs_calls[0] >= remaining:
                        return best_f
                    lbfgs_calls[0] += 1
                    return eval_func(np.clip(x, -5, 5))

                try:
                    res = minimize(
                        lbfgs_obj,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': remaining,
                            'ftol': 1e-10,
                            'gtol': 1e-8,
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 3: Final fine-tuning with tight tolerances
        if best_x is not None and call_count < self.budget:
            remaining = self.budget - call_count
            if remaining >= 5:
                final_calls = [0]

                def final_obj(x):
                    if final_calls[0] >= remaining:
                        return best_f
                    final_calls[0] += 1
                    return eval_func(np.clip(x, -5, 5))

                try:
                    res = minimize(
                        final_obj,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True,
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
