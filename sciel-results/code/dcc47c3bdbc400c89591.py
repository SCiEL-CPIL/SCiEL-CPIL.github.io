"""Multi-objective optimizer on WFG7: NSGA-II + hypervolume archive + deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II multi-objective search with a hypervolume-based archive and a wall-clock deadline."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)
        self._ref = np.array([3.0, 5.0, 7.0])

    # ------------------------------------------------------------------ #
    # hypervolume helpers (3-objective, reference point)
    # ------------------------------------------------------------------ #
    def _hv(self, pts):
        """Exact 3-objective hypervolume of points strictly better than self._ref."""
        ref = self._ref
        P = np.asarray(pts, dtype=float)
        if P.size == 0:
            return 0.0
        P = P[np.all(P < ref, axis=1)]
        n = len(P)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - P[0]))
        if n == 2:
            p, q = P[0], P[1]
            a = min(p[0], q[0]); b = max(p[0], q[0])
            c = min(p[1], q[1]); d = max(p[1], q[1])
            e = min(p[2], q[2]); f = max(p[2], q[2])
            return float((b - a) * (d - c) * (f - e)
                         + (ref[0] - b) * (ref[1] - d) * (ref[2] - f)
                         + (ref[0] - a) * (ref[1] - c) * (ref[2] - f))
        # n >= 3: sweep over objective 0
        order = np.argsort(P[:, 0])
        P = P[order]
        total = 0.0
        prev = P[0]
        for k in range(1, n):
            cur = P[k]
            total += (cur[0] - prev[0]) * self._hv2d(P[:k, 1:3])
            prev = cur
        total += (ref[0] - prev[0]) * self._hv2d(P[:, 1:3])
        return float(total)

    def _hv2d(self, Q):
        """2D hypervolume of points strictly better than ref[1:], sorted by first coord."""
        Q = np.asarray(Q, dtype=float)
        ref = self._ref[1:]
        m = len(Q)
        if m == 0:
            return 0.0
        Q = Q[np.all(Q < ref, axis=1)]
        m = len(Q)
        if m == 0:
            return 0.0
        order = np.argsort(Q[:, 0])
        Q = Q[order]
        total = 0.0
        prev = Q[0]
        for k in range(1, m):
            cur = Q[k]
            total += (cur[0] - prev[0]) * (ref[1] - prev[1])
            prev = cur
        total += (ref[0] - prev[0]) * (ref[1] - prev[1])
        return float(total)

    def _hv_archive(self, points):
        """Greedily keep the subset of points maximizing hypervolume (3-objective)."""
        pts = [np.asarray(p, dtype=float) for p in points]
        pts = [p for p in pts if np.all(p < self._ref)]
        if not pts:
            return []
        # start from the single best point
        base = max(pts, key=lambda p: float(np.prod(self._ref - p)))
        arc = [base]
        gain = self._hv(arc)
        cand = list(pts)
        while cand:
            best_i, best_gain = -1, gain
            for i, p in enumerate(cand):
                g = self._hv(arc + [p])
                if g > best_gain + 1e-12:
                    best_gain, best_i = g, i
            if best_i < 0:
                break
            arc.append(cand[best_i])
            cand.pop(best_i)
            gain = best_gain
        return arc

    # ------------------------------------------------------------------ #
    # NSGA-II helpers
    # ------------------------------------------------------------------ #
    def _fast_nondominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
            if nd[p] == 0:
                fronts[0].append(p)
                fronts[0].sort()
        i = 0
        while len(fronts[i]) > 0:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        nxt.append(q)
            if nxt:
                nxt.sort()
                fronts.append(nxt)
            i += 1
        return fronts

    def _crowding(self, F, idx):
        n = len(idx)
        if n <= 2:
            return np.full(n, 1e20)
        cd = np.zeros(n)
        Ff = F[idx]
        for m in range(self.n_obj):
            o = np.argsort(Ff[:, m])
            cd[o[0]] = 1e20
            cd[o[-1]] = 1e20
            rng = Ff[o[-1], m] - Ff[o[0], m]
            if rng <= 0:
                continue
            for k in range(1, n - 1):
                cd[o[k]] += (Ff[o[k + 1], m] - Ff[o[k - 1], m]) / rng
        return cd

    def _nsga2_step(self, X, F, pop_size, lo, hi, rng):
        n = len(F)
        fronts = self._fast_nondominated_sort(F)
        rank = np.full(n, -1, dtype=int)
        for r, fr in enumerate(fronts):
            for i in fr:
                rank[i] = r
        sel = np.argsort(rank)
        parents = [X[sel[0]], X[sel[1]]]
        c1 = parents[0] + 0.15 * (hi - lo) * rng.standard_normal(self.dim)
        c2 = parents[1] + 0.15 * (hi - lo) * rng.standard_normal(self.dim)
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        c1 = self._crossover(X[sel[0]], X[sel[1]], lo, hi, rng)
        c2 = self._crossover(X[sel[-1]], X[sel[-2]], lo, hi, rng)
        return np.array([c1, c2])

    def _crossover(self, a, b, lo, hi, rng):
        alpha = 0.5
        child = a * alpha + b * (1.0 - alpha)
        # blend with a small mutation
        child = 0.7 * child + 0.3 * (lo + (hi - lo) * rng.random(self.dim))
        return np.clip(child, lo, hi)

    # ------------------------------------------------------------------ #
    def __call__(self, problem, bounds):
        t0 = time.monotonic()
        DEADLINE = t0 + 290.0          # hard wall-clock safety margin (< 330s kill)
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        def ok():
            return time.monotonic() < DEADLINE

        # initial population
        pop_size = min(40, max(12, budget // 5))
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        all_pts = list(F)

        while used < budget and ok():
            # NSGA-II: create offspring, merge, select
            offspring = self._nsga2_step(X, F, pop_size, lo, hi, rng)
            off_F = np.array([problem(c) for c in offspring])
            used += len(offspring)
            all_pts.extend(off_F.tolist())
            if used >= budget or not ok():
                break
            # merge and select
            Xall = np.vstack([X, offspring])
            Fall = np.vstack([F, off_F])
            fronts = self._fast_nondominated_sort(Fall)
            new_idx = []
            for fr in fronts:
                if len(new_idx) + len(fr) <= pop_size:
                    new_idx.extend(fr)
                else:
                    need = pop_size - len(new_idx)
                    cd = self._crowding(Fall, fr)
                    take = np.argsort(-cd)[:need]
                    new_idx.extend([fr[t] for t in take])
                    break
            X = Xall[new_idx]
            F = Fall[new_idx]

        # final hypervolume-based archive of everything recorded
        archive = self._hv_archive(all_pts)
        return np.array(archive) if archive else np.array([F[0]])
# EVOLVE-BLOCK-END
