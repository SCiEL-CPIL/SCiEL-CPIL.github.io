"""Nelder-Mead simplex optimizer for BBOB Weierstrass f16.

Uses a single Nelder-Mead simplex run initialized at the origin to exploit
the local smoothness of the Weierstrass function for rapid early convergence,
with adaptive step sizing and no multi-start overhead.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimizer for BBOB Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0
        n_calls = [0]

        def safe(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float, copy=True)

        # ---- Nelder-Mead simplex with adaptive initial step -----------------
        start_x = np.zeros(dim)
        initial_step = 1.0

        simplex = [np.array(start_x, dtype=float)]
        fvals = [safe(start_x)]
        if np.isfinite(fvals[0]):
            record(simplex[0], fvals[0])

        for i in range(dim):
            pt = np.array(start_x, dtype=float)
            pt[i] += initial_step
            pt = np.clip(pt, -5.0, 5.0)
            simplex.append(pt)
            fvals.append(safe(pt))
            if np.isfinite(fvals[-1]):
                record(pt, fvals[-1])

        alpha, gamma, rho, sigma_c = 1.0, 2.0, 0.5, 0.5
        max_iter = budget - (dim + 1)
        stall_count = 0
        max_stall = 50

        for _ in range(max_iter):
            if n_calls[0] >= budget or time.time() > deadline:
                break

            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            if np.isfinite(fvals[0]) and np.isfinite(fvals[-1]):
                if fvals[-1] - fvals[0] < 1e-12:
                    break

            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
            fr = safe(xr)

            improved = False
            if np.isfinite(fr) and fr < fvals[0]:
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5.0, 5.0)
                fe = safe(xe)
                if np.isfinite(fe) and fe < fr:
                    simplex[-1] = xe; fvals[-1] = fe
                else:
                    simplex[-1] = xr; fvals[-1] = fr
                improved = True
            elif np.isfinite(fr) and fr < fvals[-2]:
                simplex[-1] = xr; fvals[-1] = fr
                improved = True
            else:
                if np.isfinite(fr) and fr < fvals[-1]:
                    xc = np.clip(centroid + sigma_c * (xr - centroid), -5.0, 5.0)
                else:
                    xc = np.clip(centroid - sigma_c * (centroid - simplex[-1]), -5.0, 5.0)
                fc = safe(xc)
                if np.isfinite(fc) and fc < fvals[-1]:
                    simplex[-1] = xc; fvals[-1] = fc
                    improved = True
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = safe(simplex[i])

            for x, f in zip(simplex, fvals):
                if np.isfinite(f) and f < best_f:
                    record(x, f)

            if improved:
                stall_count = 0
            else:
                stall_count += 1

            if stall_count >= max_stall:
                # Reset simplex around current best with smaller step
                stall_count = 0
                initial_step *= 0.5
                if initial_step < 1e-6:
                    break
                simplex = [best_x.copy()]
                fvals = [safe(best_x)]
                if np.isfinite(fvals[0]):
                    record(simplex[0], fvals[0])
                for i in range(dim):
                    pt = best_x.copy()
                    pt[i] += initial_step
                    pt = np.clip(pt, -5.0, 5.0)
                    simplex.append(pt)
                    fvals.append(safe(pt))
                    if np.isfinite(fvals[-1]):
                        record(pt, fvals[-1])

        return best_x, best_f
# EVOLVE-BLOCK-END
