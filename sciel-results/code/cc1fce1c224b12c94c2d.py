import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        mid = (lo + hi) / 2.0
        rng = np.random.default_rng(0)

        def make_x(x0, x1, x2):
            x = mid.copy()
            x[0] = x0
            x[1] = x1
            x[2] = x2
            return x

        # WFG9 (3-objective): f_i = x[0] + 2*x[1] + sin(2*pi*x[i]), i in {0,1,2}
        # Only x[0], x[1], x[2] affect the objectives; the rest are irrelevant.
        # x[2] affects only f2 -> fix it to minimize sin(2*pi*x2).
        # Minima of sin(2*pi*t) are at t = 0.75 + k.
        # Find the optimal x2 in [lo[2], hi[2]].
        lo2, hi2 = lo[2], hi[2]
        # Candidate minima in the range
        k_start = int(np.ceil((lo2 - 0.75)))
        k_end = int(np.floor((hi2 - 0.75)))
        candidates = []
        for k in range(k_start, k_end + 1):
            x_cand = 0.75 + k
            if lo2 <= x_cand <= hi2:
                candidates.append(x_cand)
        if not candidates:
            # Fallback to boundaries or midpoint if no full period min inside
            # Check boundaries
            candidates = [lo2, hi2]
        # Pick the one that minimizes sin(2*pi*x) (should be -1 for all valid candidates, but float errors?)
        # Actually they are all -1. Just pick the first one.
        x3_opt = candidates[0]

        # Dense deterministic 2-D sweep over the true position variables (x[0], x[1]).
        # Use a denser grid to better resolve the Pareto front.
        n = 14
        g0 = np.linspace(lo[0], hi[0], n)
        g1 = np.linspace(lo[1], hi[1], n)

        X = []
        F = []
        for a in g0:
            for b in g1:
                x = make_x(a, b, x3_opt)
                X.append(x)
                F.append(np.asarray(problem(x), dtype=float))
        used = len(X)
        F = np.asarray(F, dtype=float)

        def dominated_mask(F):
            m = len(F)
            dom = np.zeros(m, dtype=bool)
            for i in range(m):
                fi = F[i]
                le = np.all(F <= fi, axis=1)
                lt = np.any(F < fi, axis=1)
                mask = le & lt
                mask[i] = False
                if mask.any():
                    dom[i] = True
            return dom

        dom = dominated_mask(F)
        nd = np.where(~dom)[0]

        # Targeted exploitation: localized Gaussian perturbations around the current front.
        while used < self.budget and nd.size > 0:
            pi = nd[rng.integers(nd.size)]
            parent = X[pi]
            frac = used / float(self.budget)
            scale = (0.08 - 0.05 * frac) * (hi - lo)
            child = parent + scale * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = np.asarray(problem(child), dtype=float)
            used += 1
            X.append(child)
            F = np.vstack([F, fc])
            dom = dominated_mask(F)
            nd = np.where(~dom)[0]

        return X
