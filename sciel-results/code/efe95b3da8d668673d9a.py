"""Rewritten optimizer for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start optimizer for 5D BBOB-style functions.
    Strategy:
    1. Generate diverse starting points.
    2. Run L-BFGS-B from each start with limited iterations.
    3. Polish best with Nelder-Mead.
    4. Use remaining budget for (1+1)-ES refinement.
    5. Final tight polish if budget allows.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        # Generate diverse starting points
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(12):
            starts.append(rng.uniform(-5, 5, dim))
        for i in range(dim):
            for val in [-4, -2, 0, 2, 4]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        # Deduplicate
        unique_starts = []
        for s in starts:
            if not any(np.allclose(s, u, atol=0.1) for u in unique_starts):
                unique_starts.append(s)
        starts = unique_starts

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Shuffle but keep center at front
        rng.shuffle(starts)
        for i, s in enumerate(starts):
            if np.allclose(s, 0, atol=0.1):
                starts[0], starts[i] = starts[i], starts[0]
                break

        # Phase 1: Multi-start L-BFGS-B
        for s in starts:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            s = np.clip(s, -5, 5)
            try:
                res = minimize(
                    wrapped,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxls': 15, 'gtol': 1e-4, 'ftol': 1e-10}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish
        if best_x is not None and calls[0] < budget:
            try:
                remaining = budget - calls[0]
                nm_maxiter = min(80, max(10, remaining // 3))
                res_nm = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        # Phase 3: (1+1)-ES refinement
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            no_improve = 0
            max_no_improve = 40

            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1

                if no_improve > max_no_improve:
                    break
                if sigma < 1e-7:
                    break

        # Phase 4: Final tight polish
        if best_x is not None and calls[0] < budget - 5:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
