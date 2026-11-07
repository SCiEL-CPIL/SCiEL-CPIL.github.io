# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Multi-start L-BFGS-B + anisotropy-aware line-search refinement.

    L-BFGS-B's quasi-Newton Hessian approximation efficiently scales steps
    along Bent Cigar's narrow anisotropic valley, so many diverse restarts fit
    in the budget and each converges fast. A cheap finite-difference estimate of
    the valley direction followed by a bounded 1-D line search along it polishes
    the best point (Powell is avoided: it wastes calls on coordinate alignment).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0  # safety margin vs 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                xc = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(xc))
            except Exception:
                return float('inf')
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = xc.copy()
            return val

        rng = np.random.default_rng(42)

        # ---- Phase 1: diverse multi-start L-BFGS-B (global coverage) ----
        starts = [np.zeros(dim)]
        for axis in range(dim):
            for sign in (-1.0, 1.0):
                pt = np.zeros(dim)
                pt[axis] = sign * 3.0
                starts.append(pt)
        for _ in range(12):
            starts.append(rng.uniform(-4.0, 4.0, dim))
        for _ in range(8):
            starts.append(rng.uniform(-4.5, 4.5, dim))
        rng.shuffle(starts)

        n = len(starts)
        for i, x0 in enumerate(starts):
            if calls >= budget - 20 or time.time() > deadline:
                break
            remaining = budget - calls
            if i == 0:
                alloc = min(70, remaining - 10)
            elif i < 5:
                alloc = min(50, remaining - 10)
            elif i < 12:
                alloc = min(35, remaining - 10)
            else:
                alloc = min(25, remaining - 10)
            if alloc < 12:
                break
            try:
                res = scipy.optimize.minimize(
                    wrapper, x0, method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfev': alloc,
                             'ftol': 1e-12, 'gtol': 1e-8, 'eps': 1e-6},
                )
            except Exception:
                pass

        # ---- Phase 2: anisotropy-aware refinement along the valley ----
        # (bounded 1-D line search in the estimated narrow-valley direction)
        for base in (best_x, np.zeros(dim)):
            if calls >= budget - 30 or time.time() > deadline:
                break
            try:
                h = 1e-2
                f0 = wrapper(base)
                if not np.isfinite(f0):
                    continue
                d = np.zeros(dim)
                for k in range(dim):
                    xp = base.copy(); xp[k] += h
                    xm = base.copy(); xm[k] -= h
                    fp = wrapper(xp)
                    fm = wrapper(xm)
                    if np.isfinite(fp) and np.isfinite(fm):
                        d[k] = (fp - fm) / (2.0 * h)
                nrm = np.linalg.norm(d)
                if nrm < 1e-12:
                    continue
                d /= nrm
                # bounded golden-section line search along d from base
                a, b = -3.0, 3.0
                gr = (np.sqrt(5.0) - 1.0) / 2.0
                c = b - gr * (b - a)
                dpt = a + gr * (b - a)
                fc = wrapper(base + c * d)
                fd = wrapper(base + dpt * d)
                for _ in range(25):
                    if calls >= budget - 5 or time.time() > deadline:
                        break
                    if fc < fd:
                        b = dpt; dpt = c; fd = fc
                        c = b - gr * (b - a)
                        fc = wrapper(base + c * d)
                    else:
                        a = c; c = dpt; fc = fd
                        dpt = a + gr * (b - a)
                        fd = wrapper(base + dpt * d)
            except Exception:
                pass

        # ---- Deterministic fallback: never throw, always return valid point ----
        if not np.isfinite(best_f):
            best_x = np.zeros(dim)
            best_f = float('inf')
        best_x = np.clip(best_x, -5.0, 5.0)
        return best_x, best_f
# EVOLVE-BLOCK-END
