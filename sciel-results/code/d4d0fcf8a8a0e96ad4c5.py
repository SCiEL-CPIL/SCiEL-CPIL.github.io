# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective search on WFG9: NSGA-II-style selection + multi-start local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = t0 + 300.0
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        mid = 0.5 * (lo + hi)

        # --- evaluation wrapper with a global counter (hard cap at budget) ---
        count = [0]
        def ev(x):
            count[0] += 1
            return np.asarray(problem(np.asarray(x, dtype=float)), dtype=float)

        # --- deterministic multi-start candidates for local search ---
        seeds = [mid]
        for k in range(1, 9):
            seeds.append(0.5 * (lo + hi) + 0.35 * (hi - lo) * (2.0 * ((k * 0.61803398875) % 1.0) - 1.0))
        seeds = [np.clip(s, lo, hi) for s in seeds]

        # --- local search: minimize a weighted sum of objectives (Nelder-Mead) ---
        def local_search(x0, w):
            def f(x):
                return float(np.dot(w, ev(x)))
            try:
                res = minimize(f, x0, method="Nelder-Mead",
                               options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4,
                                        "adaptive": True})
            except Exception:
                return x0
            return np.clip(res.x, lo, hi)

        # --- fast non-dominance front extraction (good = low) ---
        def first_front(F):
            n = len(F)
            best = np.full(n, np.inf)
            for i in range(n):
                for j in range(n):
                    if j != i and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        best[i] = j
                        break
            return [i for i in range(n) if best[i] == np.inf]

        # --- fast crowding distance (vectorized) ---
        def crowding(F, idx):
            m = len(idx)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for k in range(n_obj):
                vals = F[idx, k]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng = vals[order[-1]] - vals[order[0]]
                if rng > 0:
                    for ii in range(1, m - 1):
                        cd[order[ii]] += (vals[order[ii + 1]] - vals[order[ii - 1]]) / rng
            return cd

        # --- binary tournament over fronts (rank then crowding) ---
        def tournament(F, idx, rank, cd, r):
            a = r.integers(len(idx))
            b = r.integers(len(idx))
            if rank[a] != rank[b]:
                return a if rank[a] < rank[b] else b
            return a if cd[a] > cd[b] else b

        # --- build initial population ---
        pop = []
        for s in seeds:
            if count[0] >= budget:
                break
            pop.append((np.asarray(s, dtype=float), ev(s)))
        r0 = np.random.default_rng(0)
        while len(pop) < 12 and count[0] < budget:
            x = np.clip(lo + r0.random(dim) * span, lo, hi)
            pop.append((x, ev(x)))
        pop = pop[:12]

        # --- initial local search phase (balanced weights) ---
        if count[0] < budget - 5:
            w_bal = np.ones(n_obj) / n_obj
            improved = []
            for (x, f) in pop:
                if count[0] >= budget - 5:
                    break
                xn = local_search(x, w_bal)
                fn = ev(xn)
                improved.append((xn, fn))
            pop = improved

        # --- main loop: NSGA-II selection + local search on chosen parents ---
        weights = [np.ones(n_obj) / n_obj,
                   np.array([0.5, 0.3, 0.2]),
                   np.array([0.2, 0.5, 0.3]),
                   np.array([0.3, 0.2, 0.5]),
                   np.array([0.5, 0.5, 0.0]),
                   np.array([0.5, 0.0, 0.5])]

        r = np.random.default_rng(1)
        while count[0] < budget and time.time() < DEADLINE:
            if len(pop) < 2:
                break
            X = np.array([p[0] for p in pop])
            F = np.array([p[1] for p in pop])
            idx = list(range(len(pop)))
            fronts = first_front(F)
            rank = np.zeros(len(pop), dtype=int)
            for d, fr in enumerate([fronts]):
                for i in fr:
                    rank[i] = 0
                remaining = [i for i in idx if i not in fr]
                rr = 1
                while remaining and rr < 50:
                    fr2 = first_front(F[remaining])
                    for i in fr2:
                        rank[remaining[i]] = rr
                    remaining = [remaining[i] for i in range(len(remaining)) if i not in fr2]
                    rr += 1
            cd = crowding(F, idx)

            # choose two parents
            pa = tournament(F, idx, rank, cd, r)
            pb = tournament(F, idx, rank, cd, r)
            parent = 0.5 * (X[pa] + X[pb])
            parent = np.clip(parent + 0.15 * span * r.standard_normal(dim), lo, hi)

            w = weights[int(r.integers(len(weights)))]
            child = local_search(parent, w)
            fc = ev(child)

            # insert: replace worst if child is not dominated by any current member
            dominated = False
            for (px, pf) in pop:
                if np.all(pf <= fc) and np.any(pf < fc):
                    dominated = True
                    break
            if not dominated:
                # remove any members dominated by the child, then replace the worst-sum
                pop = [(px, pf) for (px, pf) in pop
                       if not (np.all(fc <= pf) and np.any(fc < pf))]
                pop.append((child, fc))
                if len(pop) > 12:
                    sums = np.array([np.sum(pf) for (_, pf) in pop])
                    pop.pop(int(np.argmax(sums)))
            else:
                sums = np.array([np.sum(pf) for (_, pf) in pop])
                pop[int(np.argmax(sums))] = (child, fc)

        return X
# EVOLVE-BLOCK-END
