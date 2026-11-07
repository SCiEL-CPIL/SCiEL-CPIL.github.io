"""Optimizer for BBOB f12 (bent cigar) on [-5,5]^5.

Structural change from (1+1)-ES:
  1. Global phase: deterministic coarse 3^5 grid to locate a basin.
  2. Local phase: Nelder-Mead simplex method (derivative-free, follows the
     elongated valley with O(dim) evals per iteration).
  3. Best-of-N race: run Nelder-Mead from the top grid seeds in parallel
     (budget split), keep the best.
  4. Explicit wall-clock deadline guard.
"""
import time
import numpy as np

_DEADLINE = 330.0
_MARGIN = 3.0


class Optimizer:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        dim = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0

        calls = [0]

        def f(x):
            calls[0] += 1
            return float(func(np.clip(x, lo, hi)))

        # ---------- Global phase: 3^5 grid ----------
        grid_vals = np.linspace(lo, hi, 3)
        grid = np.array(np.meshgrid(*([grid_vals] * dim)), dtype=float)
        grid = np.stack([g.ravel() for g in grid], axis=1)  # (243, dim)
        grid_f = np.array([f(x) for x in grid])
        order = np.argsort(grid_f)

        best_x = grid[order[0]].copy()
        best_f = float(grid_f[order[0]])

        remaining = budget - calls[0]
        if remaining <= 0 or time.time() - t0 > _DEADLINE - _MARGIN:
            return best_x, best_f

        # ---------- Nelder-Mead local search ----------
        def nelder_mead(x0, f0, max_calls):
            nonlocal best_x, best_f
            # Build initial simplex around x0.
            simplex = [x0.copy()]
            values = [f0]
            delta = 0.5
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
                values.append(f(xi))

            # Sort simplex by value.
            def sort_simplex():
                idx = np.argsort(values)
                simplex[:] = [simplex[i] for i in idx]
                values[:] = [values[i] for i in idx]

            sort_simplex()

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iter = max_calls * 2  # safety; we also cap by calls/deadline

            for _ in range(max_iter):
                if calls[0] >= budget:
                    break
                if time.time() - t0 > _DEADLINE - _MARGIN:
                    break

                # Centroid of all but the worst point.
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection.
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = f(xr)
                if values[0] <= fr < values[-2]:
                    simplex[-1] = xr
                    values[-1] = fr
                    sort_simplex()
                    continue

                # Expansion.
                if fr < values[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        values[-1] = fe
                    else:
                        simplex[-1] = xr
                        values[-1] = fr
                    sort_simplex()
                    continue

                # Contraction.
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                    sort_simplex()
                    continue

                # Shrink.
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = f(simplex[i])
                sort_simplex()

            # Track best.
            if values[0] < best_f:
                best_x, best_f = simplex[0].copy(), float(values[0])
            return simplex[0], float(values[0])

        # ---------- Race: top grid seeds, budget split ----------
        K = 3
        weights = np.array([3.0, 2.0, 1.0][:K])
        weights = weights / weights.sum()
        alloc = np.maximum((remaining * weights).astype(int), 1)
        if alloc.sum() > remaining:
            scale = remaining / alloc.sum()
            alloc = np.maximum((alloc * scale).astype(int), 1)

        for k in range(K):
            if calls[0] >= budget:
                break
            if time.time() - t0 > _DEADLINE - _MARGIN:
                break
            seed_idx = order[k]
            x_seed = grid[seed_idx].copy()
            f_seed = float(grid_f[seed_idx])
            if f_seed > best_f:
                continue
            nelder_mead(x_seed, f_seed, int(alloc[k]))

        return best_x, best_f
