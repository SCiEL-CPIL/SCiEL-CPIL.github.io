# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280  # safety margin before 300s kill
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        def dominates(a, b):
            # a dominates b if a <= b in all and a < b in at least one
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            # returns list of fronts (each a list of indices)
            n = len(F)
            S = [[] for _ in range(n)]
            C = np.zeros(n, dtype=int)
            fronts = []
            f0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        C[i] += 1
                if C[i] == 0:
                    f0.append(i)
            fronts.append(f0)
            k = 0
            while fronts[k]:
                nxt = []
                for i in fronts[k]:
                    for j in S[i]:
                        C[j] -= 1
                        if C[j] == 0:
                            nxt.append(j)
                k += 1
                if nxt:
                    fronts.append(nxt)
            return fronts

        def crowding_distance(F, idx):
            # returns array of crowding distances for indices in idx
            if len(idx) <= 2:
                return np.full(len(idx), np.inf)
            Fsub = F[idx]
            m = Fsub.shape[1]
            dist = np.zeros(len(idx))
            for k in range(m):
                vals = Fsub[:, k]
                vmin, vmax = vals.min(), vals.max()
                rng_k = vmax - vmin
                if rng_k < 1e-15:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, len(order) - 1):
                    dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / rng_k
            return dist

        # Phase 1: Latin Hypercube sampling for initial coverage
        n_lhs = 40
        lhs = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            for i in range(n_lhs):
                lhs[i, d] = lo[d] + (hi[d] - lo[d]) * (perm[i] + rng.random()) / n_lhs
        lhs = np.clip(lhs, lo, hi)

        X = []
        F = []
        for x in lhs:
            if used >= self.budget:
                break
            xp, fp = eval_point(x)
            X.append(xp)
            F.append(fp)
        X = np.array(X)
        F = np.array(F)

        # Phase 2: EMOA-style loop — select parent via crowding on front, mutate
        # Keep a larger archive of non-dominated points
        archive_X = []
        archive_F = []
        for i in range(len(X)):
            archive_X.append(X[i])
            archive_F.append(F[i])
        archive_X = np.array(archive_X)
        archive_F = np.array(archive_F)

        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated set
            fronts = non_dominated_sort(archive_F)
            front0 = fronts[0]
            if len(front0) == 0:
                front0 = list(range(len(archive_F)))

            # Crowding distance on front
            cd = crowding_distance(archive_F, front0)
            # Select parent: prefer high crowding distance (diversity)
            # Use tournament of 2
            if len(front0) > 1:
                i1, i2 = rng.choice(len(front0), size=2, replace=False)
                parent_idx = front0[i1] if cd[i1] > cd[i2] else front0[i2]
            else:
                parent_idx = front0[0]

            # Mutation: adaptive step size
            # Start with larger steps, shrink over time
            progress = used / max(self.budget, 1)
            step_scale = 0.2 * (1.0 - 0.5 * progress)
            child = archive_X[parent_idx] + step_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            xp, fp = eval_point(child)
            archive_X = np.vstack([archive_X, xp])
            archive_F = np.vstack([archive_F, fp])

            # Prune archive: keep only non-dominated, limit size
            fronts = non_dominated_sort(archive_F)
            front0 = fronts[0]
            if len(front0) > 30:
                cd = crowding_distance(archive_F, front0)
                # Remove points with lowest crowding distance
                keep = set(front0)
                # Sort front0 by cd ascending, remove worst
                order = np.argsort(cd)
                to_remove = [front0[i] for i in order[:len(front0) - 30]]
                remove_set = set(to_remove)
                keep_indices = [i for i in range(len(archive_F)) if i not in remove_set]
                archive_X = archive_X[keep_indices]
                archive_F = archive_F[keep_indices]

        # Phase 3: Local polish on best second-objective point
        if used < self.budget:
            # Find point with best (lowest) second objective among those with first < 3000
            valid = [i for i in range(len(archive_F)) if archive_F[i, 0] < 3000.0]
            if valid:
                best_idx = min(valid, key=lambda i: archive_F[i, 1])
                x0 = archive_X[best_idx].copy()

                # Nelder-Mead on second objective (the binding constraint)
                def obj2(x):
                    x = np.clip(x, lo, hi)
                    f = np.asarray(problem(x), dtype=float)
                    return f[1]

                best_x = x0.copy()
                best_f2 = obj2(x0)
                used_before = used

                # Try NM with a few different initial perturbations
                for attempt in range(3):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    if attempt == 0:
                        x_init = x0.copy()
                    else:
                        x_init = x0 + 0.05 * (hi - lo) * rng.standard_normal(self.dim)
                        x_init = np.clip(x_init, lo, hi)

                    try:
                        res = scipy_minimize(
                            obj2, x_init, method='Nelder-Mead',
                            options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6}
                        )
                        if res.fun < best_f2:
                            best_f2 = res.fun
                            best_x = np.clip(res.x, lo, hi)
                    except Exception:
                        pass

                if used < self.budget:
                    xp, fp = eval_point(best_x)
                    archive_X = np.vstack([archive_X, xp])
                    archive_F = np.vstack([archive_F, fp])

        return archive_X
# EVOLVE-BLOCK-END
