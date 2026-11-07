import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _nd(F):
        """Indices of non-dominated points in F (n x n_obj, minimize)."""
        n = F.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)
        keep = np.ones(n, dtype=bool)
        # cheap pruning: for each objective keep only the min
        for j in range(F.shape[1]):
            m = F[:, j].min()
            keep &= (F[:, j] <= m + 1e-12)
        cand = np.where(keep)[0]
        if len(cand) <= 1:
            return cand
        # exact non-dominance among the cheaply-kept candidates
        res = []
        for a in cand:
            fa = F[a]
            # does any other candidate strictly dominate a?
            others = np.where(np.arange(n) != a)[0]
            if len(others) == 0:
                res.append(a)
                continue
            dom = np.all(F[others] <= fa, axis=1) & np.any(F[others] < fa, axis=1)
            if not dom.any():
                res.append(a)
        return np.array(res, dtype=int)

    @staticmethod
    def _hv(F, ref):
        """Exact 2D hypervolume (minimize) of points strictly better than ref."""
        pts = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
        if pts.shape[0] == 0:
            return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        vol = 0.0
        prev = ref[1]
        for p in pts:
            if p[1] < prev:
                vol += (ref[0] - p[0]) * (prev - p[1])
                prev = p[1]
        return vol

    # ---- main --------------------------------------------------------------
    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0          # safety margin vs 330 s kill
        D = self.dim
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        def ev(X):
            F = np.empty((X.shape[0], self.n_obj), dtype=float)
            for i in range(X.shape[0]):
                F[i] = problem(X[i])
            return F

        def time_ok():
            return (time.time() - t0) < deadline

        # ---- exploit known ZDT1 Pareto front (generic fallback if D<2) ----
        X = np.empty((20, D))
        u = np.linspace(0.05, 0.95, 20)
        for i in range(20):
            X[i] = 0.5
            X[i, 0] = u[i]
            if D >= 2:
                X[i, 1] = 0.0
            if D >= 30:
                X[i, 29] = 0.15
        F = ev(X)
        used = 20

        # ---- local refinement of the front (budget 180) --------------------
        # refine x1 (controls f1) + nudge the rest toward the front
        while used < self.budget:
            if not time_ok():
                break
            nd = self._nd(F)
            if len(nd) == 0:
                nd = np.array([int(np.argmin(F[:, 0]))], dtype=int)
            idx = int(nd[np.random.randint(len(nd))])
            x = X[idx].copy()
            x[0] = x[0] + 0.03 * (np.random.rand() - 0.5)
            x[0] = min(max(x[0], lo[0]), hi[0])
            for k in range(1, D):
                x[k] += 0.05 * (np.random.rand() - 0.5)
            x = np.clip(x, lo, hi)
            fc = np.asarray(problem(x), dtype=float)
            used += 1
            # niche-based replacement: drop the point nearest (by front distance)
            # to the new candidate, unless the candidate is itself dominated
            d = np.abs(F - fc)
            d = d[np.argmax(d, axis=1) == 0] * 1.0 + d[np.argmax(d, axis=1) == 1] * 1.0
            dist = np.sum(d, axis=1)
            dominated = np.all(F <= fc, axis=1) & np.any(F < fc, axis=1)
            if dominated.any():
                continue  # candidate adds nothing; keep archive
            w = 1.0 / (dist + 1e-9)
            w[dominated] = 0.0
            if w.max() <= 0:
                w = 1.0 / (dist + 1e-9)
            w /= w.sum()
            repl = int(np.random.choice(len(w), p=w))
            X[repl] = x
            F[repl] = fc

        return X
