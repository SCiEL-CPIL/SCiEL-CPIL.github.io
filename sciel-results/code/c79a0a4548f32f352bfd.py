# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Fast NSGA-II-style multi-objective search on WFG7 with a wall-clock guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0
        safety = 20.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng()
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng_span = hi - lo

        pop = max(8, min(60, budget // 5))
        n_evals = 0

        def time_ok():
            return (time.time() + safety) < deadline

        def evaluate(x):
            nonlocal n_evals
            f = np.asarray(problem(x), dtype=float)
            n_evals += 1
            return f

        # ---- initial population ----
        X = lo + rng_span * rng.random((pop, dim))
        F = np.empty((pop, n_obj))
        for i in range(pop):
            F[i] = evaluate(X[i])
            if not time_ok():
                break

        # ---- NSGA-II helper functions (vectorized) ----
        def non_dominated_sort(Fm):
            n = Fm.shape[0]
            front = np.empty(n, dtype=int)
            if n == 0:
                return [front]
            rank = 0
            remaining = np.arange(n)
            fronts = []
            while len(remaining) > 0:
                Fsub = Fm[remaining]
                m = len(remaining)
                dom = np.zeros(m, dtype=bool)
                for i in range(m):
                    fi = Fsub[i]
                    better = np.all(Fsub <= fi, axis=1) & np.any(Fsub < fi, axis=1)
                    better[i] = False
                    if np.any(better):
                        dom[i] = True
                non_dom = np.where(~dom)[0]
                for idx in non_dom:
                    front[remaining[idx]] = rank
                fronts.append(remaining[non_dom].copy())
                remaining = remaining[dom]
                rank += 1
            return fronts

        def crowding_distance(Fm, idx):
            m = len(idx)
            if m <= 2:
                return np.full(m, np.inf)
            Fsub = Fm[idx]
            cd = np.zeros(m)
            for j in range(Fm.shape[1]):
                vals = Fsub[:, j]
                vmin = vals.min()
                vmax = vals.max()
                if vmax <= vmin:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                cd[1:-1] += (vals[order[2:]] - vals[order[:-2]]) / (vmax - vmin)
            return cd

        # ---- main loop ----
        while n_evals < budget and time_ok():
            fronts = non_dominated_sort(F)
            new_X = []
            new_F = []
            for f in fronts:
                if len(new_X) + len(f) <= pop:
                    new_X.append(X[f])
                    new_F.append(F[f])
                else:
                    room = pop - len(new_X)
                    cd = crowding_distance(F, f)
                    keep = np.argsort(-cd)[:room]
                    new_X.append(X[f[keep]])
                    new_F.append(F[f[keep]])
                    break
            X = np.vstack(new_X) if new_X else X
            F = np.vstack(new_F) if new_F else F

            # ---- mating: tournament + SBX + polynomial mutation ----
            n_child = 2 * (pop - len(X))
            if n_child <= 0:
                break
            p1 = np.empty((n_child, dim))
            p2 = np.empty((n_child, dim))
            for k in range(n_child):
                c1 = rng.integers(pop)
                c2 = rng.integers(pop)
                p1[k] = X[c1]
                p2[k] = X[c2]

            # Simulated binary crossover
            eta_c = 20.0
            children = np.empty((n_child, dim))
            for k in range(n_child):
                a = p1[k]
                b = p2[k]
                child = a.copy()
                for d in range(dim):
                    if rng.random() < 0.5:
                        if a[d] != b[d]:
                            r = rng.random()
                            if r <= 0.5:
                                beta = (2.0 * r) ** (1.0 / (eta_c + 1.0))
                            else:
                                beta = (1.0 / (2.0 * (1.0 - r))) ** (1.0 / (eta_c + 1.0))
                            child[d] = 0.5 * ((a[d] + b[d]) - beta * (b[d] - a[d]))
                children[k] = child

            # Polynomial mutation
            eta_m = 20.0
            for k in range(n_child):
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        span = hi[d] - lo[d]
                        r = rng.random()
                        if r < 0.5:
                            delta = (2.0 * r) ** (1.0 / (eta_m + 1.0)) - 1.0
                        else:
                            delta = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                        children[k, d] += delta * span
            children = np.clip(children, lo, hi)

            # ---- evaluate children ----
            child_F = np.empty((n_child, n_obj))
            for k in range(n_child):
                child_F[k] = evaluate(children[k])
                if not time_ok():
                    break

            # ---- combine and select ----
            allX = np.vstack([X, children])
            allF = np.vstack([F, child_F])
            fronts = non_dominated_sort(allF)
            sel = np.empty(pop, dtype=int)
            nsel = 0
            for f in fronts:
                if nsel + len(f) <= pop:
                    sel[nsel:nsel + len(f)] = f
                    nsel += len(f)
                else:
                    room = pop - nsel
                    cd = crowding_distance(allF, f)
                    keep = np.argsort(-cd)[:room]
                    sel[nsel:nsel + room] = f[keep]
                    nsel += room
                    break
            X = allX[sel]
            F = allF[sel]

        return X
# EVOLVE-BLOCK-END
