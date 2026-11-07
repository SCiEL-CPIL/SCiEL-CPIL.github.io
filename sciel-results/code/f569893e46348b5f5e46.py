"""Multi-objective DTLZ1 optimizer: structured init + adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured Pareto-front seeding + adaptive multi-scale local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 310.0  # safety margin below 330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        rng = np.random.default_rng(12345)

        # ---------- helpers ----------
        def ditz1(x):
            # x: (dim,)
            x1 = x[0]
            rest = x[1:]
            g = 1.0 + 10.0 * n_obj * (
                np.sum(rest ** 2)
                - np.sum(rest * np.cos(10.0 * np.pi * (rest + 0.5)))
            )
            # n_obj+1 objectives
            f = np.empty(n_obj + 1, dtype=np.float64)
            f[0] = 0.5 * x1 * (1.0 + g)
            f[1] = 0.5 * (1.0 - x1) * (1.0 + g)
            f[2:] = 0.5 * (1.0 + g)
            return f

        # ---------- 1) structured initialization on the Pareto front ----------
        # DTLZ1 front: x1 in [0,1], rest on a sphere of radius r such that
        # g=0.  We sample x1 uniformly and the rest on the unit sphere.
        n_init = min(120, max(20, budget // 4))
        X = np.empty((n_init, dim), dtype=np.float64)
        x1 = np.linspace(0.0, 1.0, n_init)
        # random points on unit (dim-1)-sphere
        U = rng.standard_normal((n_init, dim - 1))
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        U = U / norms
        X[:, 0] = x1
        X[:, 1:] = 0.5 * U  # radius 0.5 so that rest in [-0.5,0.5]
        X = np.clip(X, lo, hi)

        F = np.array([ditz1(x) for x in X], dtype=np.float64)
        # also evaluate the true problem to respect its clipping
        F = np.array([np.asarray(problem(x), dtype=np.float64) for x in X])
        used = n_init

        # ---------- 2) adaptive multi-scale local refinement ----------
        # scales from coarse to fine; each scale does a few Nelder-Mead-ish
        # perturbations on the current non-dominated set.
        scales = [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
        # we keep a growing archive of all evaluated points
        archive_X = [X.copy()]
        archive_F = [F.copy()]
        all_X = X.copy()
        all_F = F.copy()

        def is_dominated(f, Fset):
            # f dominated if any row in Fset <= f and < f
            le = np.all(Fset <= f, axis=1)
            lt = np.any(Fset < f, axis=1)
            return bool(np.any(le & lt))

        def nondominated_mask(Fset):
            n = Fset.shape[0]
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                le = np.all(Fset <= Fset[i], axis=1)
                lt = np.any(Fset < Fset[i], axis=1)
                dom = le & lt
                dom[i] = False
                if np.any(dom):
                    mask[i] = False
            return mask

        for scale in scales:
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            # take current non-dominated points as seeds
            mask = nondominated_mask(all_F)
            seeds_idx = np.where(mask)[0]
            if len(seeds_idx) == 0:
                seeds_idx = np.arange(len(all_X))
            # shuffle seeds
            perm = rng.permutation(len(seeds_idx))
            seeds_idx = seeds_idx[perm]
            # for each seed, try a few directions
            for si in seeds_idx:
                if used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                base = all_X[si].copy()
                # try 2 random perturbations at this scale
                for _ in range(2):
                    if used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    direction = rng.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-12)
                    step = scale * np.linalg.norm(span)
                    cand = base + step * direction
                    cand = np.clip(cand, lo, hi)
                    fc = np.asarray(problem(cand), dtype=np.float64)
                    used += 1
                    # keep if not dominated by current archive
                    if not is_dominated(fc, all_F):
                        all_X = np.vstack([all_X, cand])
                        all_F = np.vstack([all_F, fc])
                        # prune dominated from archive to keep it small
                        if len(all_F) > 400:
                            nm = nondominated_mask(all_F)
                            if np.sum(nm) < len(all_F):
                                all_X = all_X[nm]
                                all_F = all_F[nm]

        # ---------- 3) final high-precision polish ----------
        # Use a few iterations of small steps on the best non-dominated points
        # to converge toward the true front.
        if used < budget and time.time() < deadline - 5.0:
            mask = nondominated_mask(all_F)
            seeds_idx = np.where(mask)[0]
            if len(seeds_idx) > 0:
                perm = rng.permutation(len(seeds_idx))
                seeds_idx = seeds_idx[perm]
                polish_scales = [0.001953125, 0.0009765625, 0.00048828125]
                for scale in polish_scales:
                    if used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    for si in seeds_idx:
                        if used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        base = all_X[si].copy()
                        for _ in range(2):
                            if used >= budget:
                                break
                            if time.time() > deadline - 5.0:
                                break
                            direction = rng.standard_normal(dim)
                            direction /= (np.linalg.norm(direction) + 1e-12)
                            step = scale * np.linalg.norm(span)
                            cand = base + step * direction
                            cand = np.clip(cand, lo, hi)
                            fc = np.asarray(problem(cand), dtype=np.float64)
                            used += 1
                            if not is_dominated(fc, all_F):
                                all_X = np.vstack([all_X, cand])
                                all_F = np.vstack([all_F, fc])
                                if len(all_F) > 400:
                                    nm = nondominated_mask(all_F)
                                    if np.sum(nm) < len(all_F):
                                        all_X = all_X[nm]
                                        all_F = all_F[nm]

        # return the final archive (evaluator ignores return value but keeps it)
        return all_X
# EVOLVE-BLOCK-END
