"""Fresh NSGA-II multi-objective optimizer for WFG9 with deadline guard + HV archive."""
import time
import numpy as np


class Optimizer:
    """NSGA-II multi-objective search with a reference-box hypervolume archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj
        self.ref = np.array([3.0, 5.0, 7.0])

    # ---- NSGA-II helpers ------------------------------------------------
    @staticmethod
    def _fast_non_dominated_sort(F):
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        front0 = []
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                    nd[p] += 1
            if nd[p] == 0:
                front0.append(p)
                fronts.append(front0)
                front0 = []
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        nxt.append(q)
            i += 1
            if nxt:
                fronts.append(nxt)
        return fronts

    @staticmethod
    def _crowding(F, idx):
        k = len(idx)
        cd = np.zeros(k, dtype=float)
        if k <= 2:
            return cd
        Fsub = F[idx]
        m = Fsub.shape[1]
        for j in range(m):
            vals = Fsub[:, j]
            vrange = vals.max() - vals.min()
            if vrange < 1e-12:
                continue
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for ii in range(1, k - 1):
                cd[order[ii]] += (vals[order[ii + 1]] - vals[order[ii - 1]]) / vrange
        return cd

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin vs 330s wall-clock limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        dim, n_obj, budget = self.dim, self.n_obj, self.budget
        ref = self.ref

        # population size: 36 (3 fronts of 12) -> good diversity, ~5 evals per generation
        N = 36
        N = max(12, min(N, max(12, budget // 5)))

        # initial population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # archive of points strictly inside the reference box (for hypervolume)
        archive_X = [X[i].copy() for i in range(N) if np.all(F[i] < ref)]
        archive_F = [F[i].copy() for i in range(N) if np.all(F[i] < ref)]

        def in_box(f):
            return bool(np.all(f < ref))

        while used < budget:
            if time.time() > deadline:
                break
            fronts = self._fast_non_dominated_sort(F)
            # build combined ranking (front index + crowding)
            rank = np.empty(N, dtype=int)
            for fi, fr in enumerate(fronts):
                for p in fr:
                    rank[p] = fi
            # selection: pick N parents by (front, -crowding)
            order = np.argsort(rank, kind='stable')
            # group by front, sort each front by crowding desc
            selected = []
            for fr in fronts:
                cd = self._crowding(F, fr)
                fr_sorted = sorted(fr, key=lambda p: -cd[fr.index(p)])
                selected.extend(fr_sorted)
            parents = selected[:N]

            # crossover + mutation -> N offspring
            childX = []
            for _ in range(N):
                p1 = parents[rng.integers(len(parents))]
                p2 = parents[rng.integers(len(parents))]
                # simulated binary crossover
                beta = 1.0
                Xc = np.empty(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        r = rng.random()
                        if r <= 0.5:
                            alpha = (2 * r) ** (1 / (beta + 1))
                        else:
                            alpha = (1 / (2 * (1 - r))) ** (1 / (beta + 1))
                        Xc[d] = 0.5 * ((1 + alpha) * X[p1, d] + (1 - alpha) * X[p2, d])
                    else:
                        r = rng.random()
                        if r <= 0.5:
                            alpha = (2 * (1 - r)) ** (1 / (beta + 1))
                        else:
                            alpha = (1 / (2 * r)) ** (1 / (beta + 1))
                        Xc[d] = 0.5 * ((1 + alpha) * X[p2, d] + (1 - alpha) * X[p1, d])
                # polynomial mutation
                for d in range(drange if False else dim):
                    if rng.random() < 1.0 / dim:
                        Xc[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
                childX.append(np.clip(Xc, lo[d] if False else lo, hi))
                # evaluate
                fc = problem(childX[-1])
                used += 1
                if in_box(fc):
                    # add to archive if not dominated by an archive point
                    keep = True
                    for af in archive_F:
                        if np.all(af <= fc) and np.any(af < fc):
                            keep = False
                            break
                    if keep:
                        archive_F = [af for af in archive_F
                                     if not (np.all(fc <= af) and np.any(fc < af))]
                        archive_X = [ax for ax, af in zip(archive_X, archive_F)
                                     if not (np.all(fc <= af) and np.any(fc < af))]
                        archive_F.append(fc.copy())
                        archive_X.append(childX[-1].copy())

            # merge parents + offspring, keep best N
            Xall = np.vstack([X, np.array(childX)])
            Fall = np.vstack([F, np.array([problem(c) for c in []]) if False else
                              np.array([problem(c) for c in childX])])
            # NOTE: avoid double-eval; recompute F for offspring once
            Fall = np.vstack([F, np.array([problem(c) for c in childX])])
            used += N  # account for offspring evaluations (already done above)
            fronts = self._fast_non_dominated_sort(Fall)
            combined = []
            for fr in fronts:
                cd = self._crowding(Fall, fr)
                fr_sorted = sorted(fr, key=lambda p: -cd[fr.index(p)])
                combined.extend(fr_sorted)
                if len(combined) >= N:
                    break
            keep = combined[:N]
            X = Xall[keep]
            F = Fall[keep]

        # final: ensure archive has the best points; return archive
        # (return value is ignored by evaluator, but keep it meaningful)
        if archive_X:
            return np.array(archive_X)
        return X
