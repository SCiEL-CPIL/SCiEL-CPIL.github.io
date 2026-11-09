"""Island-model multi-objective optimizer: hypervolume-driven, weight-partitioned."""
import time
import numpy as np


class Optimizer:
    """Island model: objective-weight sub-regions, each with a small population
    and weight-specific mutation intensity. Hypervolume drives replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._ref = np.array([3.0, 5.0, 7.0], dtype=float)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0          # safety margin under 300 s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        dim, n_obj, budget = self.dim, self.n_obj, self.budget

        # ---- objective weights: vertices + edge midpoints of the simplex ----
        W = [np.eye(n_obj)[i] for i in range(n_obj)]
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                w = np.zeros(n_obj); w[i] = 0.5; w[j] = 0.5
                W.append(w)
        W = np.array(W, dtype=float)
        n_islands = len(W)
        n_islands = max(1, min(n_islands, budget // 6))

        # ---- per-island mutation intensity: small for sharp corners, large for broad midpoints ----
        conc = W.sum(1) ** 2            # 1 for corners, 0.5 for midpoints
        mu = np.linspace(0.03, 0.25, n_islands)
        mu = np.clip(mu, 0.03, 0.30)

        # ---- population size per island (keep total init <= budget) ----
        per = max(3, budget // (4 * n_islands))
        per = min(per, 24)

        # ---- initialize: deterministic spread per island + one random ----
        X = []
        for i in range(n_islands):
            base = lo + (hi - lo) * rng.random(dim)
            for k in range(per):
                X.append(np.clip(base + mu[i] * (hi - lo) * rng.standard_normal(dim), lo, hi))
        X = np.array(X)
        F = np.array([problem(x) for x in X])
        used = len(X)
        shapes = [per] * n_islands
        idx = 0

        def hv_contribution(Fs):
            """Contribution of each point in Fs to the union hypervolume (vs ref)."""
            n = len(Fs)
            if n == 0:
                return np.zeros(0)
            valid = np.all(Fs < self._ref, axis=1)
            if not np.any(valid):
                return np.zeros(n)
            G = Fs[valid]
            if len(G) == 1:
            # single point: product of (ref - f)
                c = np.zeros(n)
                c[valid] = np.prod(self._ref - G)
                return c
            # sort by first objective descending (worse first)
            order = np.argsort(-G[:, 0])
            Gs = G[order]
            best = self._ref.copy()
            contrib = np.zeros(len(Gs))
            for k in range(len(Gs) - 1, -1, -1):
                p = Gs[k]
                box = np.maximum(best - p, 0.0)
                contrib[k] = np.prod(box)
                best = np.minimum(best, p)
            c = np.zeros(n)
            c[valid] = contrib
            c[order] = c[valid]  # reorder back to valid order
            c = np.zeros(n)
            c[valid] = contrib
            return c

        # ---- main loop: one island per step, replace worst contributor ----
        step = 0
        while used < budget:
            if time.time() > deadline:
                break
            i = step % n_islands
            start = sum(shapes[:i]); end = start + shapes[i]
            Fi = F[start:end]
            Xi = X[start:end]
            # pick a parent: prefer low weighted sum, with some randomness
            ws = Fi @ W[i]
            # tournament of 2 by weighted sum
            a, b = rng.integers(end - start), rng.integers(end - start)
            parent = Xi[a] if ws[a] <= ws[b] else Xi[b]
            child = np.clip(parent + mu[i] * (hi - lo) * rng.standard_normal(dim), lo, hi)
            fc = problem(child)
            used += 1
            # evaluate contribution of the candidate set (current island + child)
            cand_F = np.vstack([Fi, fc.reshape(1, -1)])
            cand_X = np.vstack([Xi, child.reshape(1, -1)])
            contrib = hv_contribution(cand_F)
            # drop the point with the smallest contribution (ties -> last)
            drop = int(np.argmin(contrib))
            if drop == len(Fi):
                # child is worst; keep child only if it's not dominated by all
                # still replace the worst existing to keep population fresh
                drop = int(np.argmin(contrib[:len(Fi)]))
            X[start + drop] = child
            F[start + drop] = fc
            step += 1

        return F
