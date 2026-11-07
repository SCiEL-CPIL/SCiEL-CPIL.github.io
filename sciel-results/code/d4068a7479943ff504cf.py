"""Hybrid global + local optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Dense global sampling to find the basin, then Nelder-Mead + (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280.0
        dim = self.dim
        n = self.budget
        calls = 0

        def safe(x):
            nonlocal calls
            calls += 1
            return float(func(np.clip(np.asarray(x, dtype=float), -5.0, 5.0)))

        # ---- Phase 1: dense Latin Hypercube global sampling ----
        n_lhs = min(200, n)
        perm = np.random.permutation(n_lhs)
        u = (np.arange(n_lhs) + 0.5 + (perm - (n_lhs - 1) / 2.0) / n_lhs) / n_lhs
        X = u * 10.0 - 5.0  # shape (n_lhs, dim)
        vals = np.empty(n_lhs)
        for i in range(n_lhs):
            if calls >= n:
                break
            vals[i] = safe(X[i])
        valid = np.where(np.isfinite(vals))[0]
        if len(valid) == 0:
            return X[0], vals[0]
        best_i = valid[np.argmin(vals[valid])]
        x = X[best_i].copy()
        fx = vals[best_i]
        best_x, best_fx = x.copy(), fx

        # ---- Phase 2: Nelder-Mead on the best basin ----
        def nelder_mead(x0, f0, max_iters, max_calls):
            nonlocal calls, best_x, best_fx
            dim = self.dim
            # Build simplex
            simplex = [x0.copy()]
            for k in range(dim):
                pt = x0.copy()
                pt[k] += 0.1 if pt[k] < 4.9 else -0.1
                simplex.append(pt)
            fvals = [f0]
            for k in range(1, dim + 1):
                if calls >= n or calls >= max_calls:
                    break
                fvals.append(safe(simplex[k]))
            # Iterate
            for _ in range(max_iters):
                if calls >= n or calls >= max_calls:
                    break
                if time.time() > deadline:
                    break
                order = np.argsort(fvals)
                simplex = [simplex[j] for j in order]
                fvals = [fvals[j] for j in order]
                # Track best
                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()
                worst = dim
                # Reflection
                centroid = np.mean(simplex[:dim], axis=0)
                xr = centroid + 1.0 * (centroid - simplex[worst])
                fr = safe(xr)
                if fvals[0] <= fr < fvals[worst - 1]:
                    simplex[worst] = xr
                    fvals[worst] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (xr - centroid)
                    fe = safe(xe)
                    if fe < fr:
                        simplex[worst] = xe
                        fvals[worst] = fe
                    else:
                        simplex[worst] = xr
                        fvals[worst] = fr
                else:
                    # Contraction
                    if fr < fvals[worst]:
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        xc = centroid - 0.5 * (simplex[worst] - centroid)
                    fc = safe(xc)
                    if fc < min(fr, fvals[worst]):
                        simplex[worst] = xc
                        fvals[worst] = fc
                    else:
                        # Shrink
                        for k in range(1, dim + 1):
                            simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                            if calls >= n or calls >= max_calls:
                                break
                            fvals[k] = safe(simplex[k])
                        if fvals[0] < best_fx:
                            best_fx = fvals[0]
                            best_x = simplex[0].copy()
            # Update best from simplex
            if len(fvals) > 0:
                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()
            return simplex[0].copy(), fvals[0]

        nm_iters = 120
        nm_calls = min(150, max(0, n - calls))
        if nm_calls > 5:
            x, fx = nelder_mead(x, fx, nm_iters, nm_calls)

        # ---- Phase 3: (1+1)-ES polish with adaptive step ----
        rng = np.random.default_rng(1)
        sigma = 0.5
        x = best_x.copy()
        fx = best_fx
        for _ in range(n - calls):
            if calls >= n:
                break
            if time.time() > deadline:
                break
            y = x + sigma * rng.standard_normal(dim)
            fy = safe(y)
            if np.isfinite(fy) and fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.8
        return best_x, best_fx
# EVOLVE-BLOCK-END
