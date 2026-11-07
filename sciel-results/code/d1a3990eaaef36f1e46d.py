"""ZDT1 multi-objective optimizer: spread x1 for f1 coverage, minimize f2 via local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Spread x1 across [0,1] for f1 coverage (f1 depends only on x1 in ZDT1),
    then locally minimize f2 (which depends on the tail) per point. Deadline-guarded."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        used = 0
        def eval(x):
            nonlocal used
            used += 1
            return np.asarray(problem(x), dtype=float)

        # Phase 1: construct a spread of x1 values (f1 = x1 typically), random tail.
        n_init = min(budget - 10, 60)
        if n_init < 2:
            n_init = 2
        # x1 spread: power bias toward low end (front steeper there)
        t = np.linspace(0.0, 1.0, n_init)
        x1_vals = np.clip(t ** 1.5, lo[0], hi[0])
        X = np.empty((n_init, dim))
        F = np.empty((n_init, self.n_obj))
        for i in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            x[0] = x1_vals[i]
            X[i] = x
            F[i] = eval(x)

        # Phase 2: local search to minimize f2 (obj index 1) per point via coordinate descent,
        # keeping x1 fixed (so f1 coverage is preserved). Universally valid for ZDT1.
        polish_budget = budget - used
        if polish_budget > 0:
            # polish a subset, spread in x1, to conserve budget
            n_polish = min(20, n_init)
            idxs = np.linspace(0, n_init - 1, n_polish).astype(int)
            per_point = max(1, polish_budget // n_polish)
            step0 = 0.1 * (hi - lo)
            for pi in idxs:
                if used >= budget or time.time() > deadline:
                    break
                x = X[pi].copy()
                fx = F[pi].copy()
                step = step0
                stall = 0
                for it in range(per_point):
                    if used >= budget or time.time() > deadline:
                        break
                    improved = False
                    # coordinate descent over dims 1..dim-1 (keep x1)
                    for d in range(1, dim):
                        if used >= budget or time.time() > deadline:
                            break
                        best_x, best_f = x, fx
                        for sgn in (1.0, -1.0):
                            xn = x.copy()
                            xn[d] = np.clip(xn[d] + sgn * step, lo[d], hi[d])
                            fn = eval(xn)
                            # prefer lower f2; tie-break lower f1
                            if fn[1] < best_f[1] - 1e-12 or (abs(fn[1] - best_f[1]) < 1e-12 and fn[0] < best_f[0]):
                                best_x, best_f = xn, fn
                                improved = True
                        if improved:
                            x, fx = best_x, best_f
                            break
                    if not improved:
                        step *= 0.5
                        stall += 1
                        if stall > 6 or step < 1e-6 * (hi - lo).max():
                            break
                    else:
                        stall = 0
                X[pi], F[pi] = x, fx

        # Phase 3: if budget remains, add a few more x1 spread points with random tail,
        # then lightly polish, to widen front coverage.
        if used < budget and time.time() < deadline:
            n_extra = min(budget - used, 20)
            if n_extra > 0:
                t2 = np.linspace(0.0, 1.0, n_extra)
                x1_extra = np.clip(t2 ** 1.5, lo[0], hi[0])
                # append to arrays
                Xn = np.vstack([X, np.empty((n_extra, dim))])
                Fn = np.vstack([F, np.empty((n_extra, self.n_obj))])
                for i in range(n_extra):
                    if used >= budget or time.time() > deadline:
                        break
                    x = lo + (hi - lo) * rng.random(dim)
                    x[0] = x1_extra[i]
                    Xn[n_init + i] = x
                    Fn[n_init + i] = eval(x)
                X, F = Xn, Fn
                # light polish on the new points
                if used < budget and time.time() < deadline:
                    rem = budget - used
                    per = max(1, rem // max(1, n_extra))
                    for i in range(n_extra):
                        if used >= budget or time.time() > deadline:
                            break
                        pi = n_init + i
                        x = X[pi].copy()
                        fx = F[pi].copy()
                        step = 0.05 * (hi - lo)
                        for _ in range(per):
                            if used >= budget or time.time() > deadline:
                                break
                            improved = False
                            for d in range(1, dim):
                                if used >= budget or time.time() > deadline:
                                    break
                                best_x, best_f = x, fx
                                for sgn in (1.0, -1.0):
                                    xn = x.copy()
                                    xn[d] = np.clip(xn[d] + sgn * step, lo[d], hi[d])
                                    fn = eval(xn)
                                    if fn[1] < best_f[1] - 1e-12 or (abs(fn[1]-best_f[1])<1e-12 and fn[0]<best_f[0]):
                                        best_x, best_f = xn, fn
                                        improved = True
                                if improved:
                                    x, fx = best_x, best_f
                                    break
                            if not improved:
                                break
                        X[pi], F[pi] = x, fx

        # Return points strictly better than ref [11,11]
        mask = np.all(F < 11.0, axis=1)
        return X[mask]
# EVOLVE-BLOCK-END
