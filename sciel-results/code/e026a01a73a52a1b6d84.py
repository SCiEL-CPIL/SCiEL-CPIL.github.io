# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast vectorized NSGA-II for WFG9 (3 objectives, 12 dims)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _dominates(F):
        # D[i, j] True if i dominates j (i <= j all, i < j some)
        return (F[:, None, :] <= F[None, :, :]) & (F[:, None, :] < F[None, :, :]).any(axis=2)

    def _rank(self, F):
        n = F.shape[0]
        rank = np.full(n, -1, dtype=int)
        remaining = np.arange(n)
        r = 0
        while remaining.size:
            D = self._dominates(F[remaining])
            front = remaining[~D.any(axis=1)]
            rank[front] = r
            remaining = remaining[D.any(axis=1)]
            r += 1
        return rank

    def _crowd(self, F):
        n = F.shape[0]
        cd = np.zeros(n)
        if n <= 1:
            return cd
        for k in range(F.shape[1]):
            mn, mx = F[:, k].min(), F[:, k].max()
            rng = mx - mn
            if rng < 1e-12:
                continue
            order = np.argsort(F[:, k])
            cd[order[0]] += 1.0
            cd[order[-1]] += 1.0
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / rng
        return cd

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin under 300s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random  # evaluator sets np.random.seed(seed) before calling us

        pop = min(50, max(10, self.budget // 4))
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop

        while used < self.budget and time.time() < deadline:
            # ---- selection: fast non-dominated sort + crowding ----
            rank = self._rank(F)
            order = np.argsort(rank)
            Fsort = F[order]
            cd = self._crowd(Fsort)
            selected = order[np.argsort(-cd)][:pop]

            # ---- reproduction: crossover + mutation ----
            parents = X[selected]
            p1 = parents[rng.integers(pop)]
            p2 = parents[rng.integers(pop)]
            mask = rng.random(self.dim) < 0.5
            child = np.where(mask, p1, p2)
            sigma = 0.2
            mut = (rng.random(self.dim) < 0.15)
            child = child + sigma * (hi - lo) * rng.standard_normal(self.dim) * mut
            child = np.clip(child, lo, hi)
            fc = np.asarray(problem(child), dtype=float)
            used += 1

            # ---- elitist replacement: keep best pop of (parents + child) ----
            FX = np.vstack([F, fc])
            Xall = np.vstack([X, child])
            rank = self._rank(FX)
            order = np.argsort(rank)
            Fsort = FX[order]
            cd = self._crowd(Fsort)
            keep = order[np.argsort(-cd)][:pop]
            X = Xall[keep]
            F = FX[keep]

        return X
# EVOLVE-BLOCK-END
