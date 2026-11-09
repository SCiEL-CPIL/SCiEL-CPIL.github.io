# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Fast NSGA-II multi-objective optimizer for WFG4 with a wall-clock guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Return (fronts, rank, ndcount) via fast non-dominated sorting."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        rank = np.full(n, -1, dtype=int)
        fronts = []
        cur = []
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
            if nd[p] == 0:
                rank[p] = 0
                cur.append(p)
        i = 0
        while cur:
            fronts.append(cur)
            nxt = []
            for p in cur:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        rank[q] = i + 1
                        nxt.append(q)
            cur = nxt
            i += 1
        return fronts, rank, nd

    def _crowding(self, F, front):
        """Assign crowding distance to members of a single front."""
        m = len(front)
        dist = np.zeros(m)
        if m <= 2:
            dist[:] = np.inf
            return dist
        for k in range(self.n_obj):
            vals = np.array([F[i, k] for i in front])
            order = np.argsort(vals)
            fmin = vals[order[0]]
            fmax = vals[order[-1]]
            rng = fmax - fmin
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            if rng > 0:
                for j in range(1, m - 1):
                    dist[order[j]] += (vals[order[j + 1]] - vals[order[j - 1]]) / rng
        return dist

    def _evaluate(self, problem, X):
        """Evaluate a batch; falls back to row-by-row if the problem is scalar-only."""
        try:
            F = problem(X)
            F = np.asarray(F, dtype=float)
            if F.ndim == 1:
                F = F.reshape(1, -1)
            if F.shape[0] != len(X):
                raise ValueError("shape mismatch")
            if not np.all(np.isfinite(F)):
                raise ValueError("non-finite")
            return F
        except Exception:
            return np.array([np.asarray(problem(x), dtype=float) for x in X])

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 250.0  # safety margin well under the 330s wall-clock kill
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        pop = int(min(30, max(8, budget // 4)))
        X = lo + (hi - lo) * rng.random((pop, dim))
        F = self._evaluate(problem, X)
        used = pop

        # keep every recorded objective vector (deduplicated) for the hypervolume
        rec = []
        for f in F:
            rec.append(tuple(f))
        recset = set(rec)

        def add_record(f):
            t = tuple(f)
            if t not in recset:
                recset.add(t)
                rec.append(t)

        def make_child():
            # binary tournament on (rank, crowding)
            i1, i2 = rng.integers(pop, size=2)
            if rank[i1] != rank[i2]:
                p1 = i1 if rank[i1] < rank[i2] else i2
            else:
                p1 = i1 if crowd[i1] > crowd[i2] else i2
            j1, j2 = rng.integers(pop, size=2)
            if rank[j1] != rank[j2]:
                p2 = j1 if rank[j1] < rank[j2] else j2
            else:
                p2 = j1 if crowd[j1] > crowd[j2] else j2
            # uniform crossover
            mask = rng.random(dim) < 0.5
            child = np.where(mask, X[p1], X[p2])
            # adaptive mutation
            step = 0.35 * (hi - lo) * (0.15 + 0.85 * (1.0 - used / max(1, budget)))
            child = child + step * rng.standard_normal(dim)
            return np.clip(child, lo, hi)

        while used < budget:
            if time.time() > deadline:
                break
            fronts, rank, nd = self._non_dominated_sort(F)
            crowd = np.zeros(pop)
            for fr in fronts:
                cd = self._crowding(F, fr)
                for a, b in zip(fr, cd):
                    crowd[a] = b

            # create a batch of children (one generation)
            gen = pop
            while gen > 0 and used < budget and time.time() < deadline:
                b = min(gen, budget - used)
                batch = np.array([make_child() for _ in range(b)])
                bf = self._evaluate(problem, batch)
                used += b
                gen -= b
                for x, f in zip(batch, bf):
                    X[used - b + (x is x)]  # no-op guard
                # replace worst-by-rank/crowding parents one-for-one with the batch
                for idx in range(b):
                    # pick the individual with the worst (rank, crowd) to replace
                    order = np.lexsort(([-crowd[i] for i in range(pop)],
                                       [rank[i] for i in range(pop)]))
                    w = order[-1]
                    X[w] = batch[idx]
                    F[w] = bf[idx]
                    add_record(bf[idx])
                    # recompute cheap proxy for crowd after replacement is done at loop top

        return X
# EVOLVE-BLOCK-END
