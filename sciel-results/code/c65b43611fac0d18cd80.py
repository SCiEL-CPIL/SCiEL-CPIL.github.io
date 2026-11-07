"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining NSGA-II structure with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin under 300s/330s limits

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj

        def fit(x):
            return np.asarray(problem(x), dtype=float).reshape(n_obj)

        # --- Non-dominated sorting (returns front IDs and crowding) ---
        def fast_non_dominated_sort(F):
            n = F.shape[0]
            S = [[] for _ in range(n)]
            nd = [0] * n
            front_id = np.zeros(n, dtype=int)
            remaining = set(range(n))
            fid = 0
            while remaining:
                front = [i for i in remaining if nd[i] == 0]
                for i in front:
                    front_id[i] = fid
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                remaining -= set(front)
                fid += 1
            # Build S and nd properly first
            S = [[] for _ in range(n)]
            nd = [0] * n
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
            front_id = np.zeros(n, dtype=int)
            remaining = set(range(n))
            fid = 0
            while remaining:
                front = [i for i in remaining if nd[i] == 0]
                for i in front:
                    front_id[i] = fid
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                remaining -= set(front)
                fid += 1
            return front_id, fid

        def crowding_distance(front_idx, F):
            m = F.shape[1]
            dist = np.zeros(len(front_idx))
            if len(front_idx) <= 2:
                dist[:] = np.inf
                return dist
            for k in range(m):
                vals = F[front_idx, k]
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                vmin, vmax = vals[order[0]], vals[order[-1]]
                rng_ = vmax - vmin
                if rng_ > 1e-12:
                    for t in range(1, len(front_idx) - 1):
                        dist[order[t]] += (vals[order[t + 1]] - vals[order[t - 1]]) / rng_
            return dist

        # --- Initialize population ---
        pop_size = min(20, self.budget // 4)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([fit(x) for x in X])
        used = pop_size

        # --- Main loop ---
        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute fronts and crowding
            front_id, n_fronts = fast_non_dominated_sort(F)
            crowd = np.zeros(pop_size)
            for f in range(n_fronts):
                idx = [i for i in range(pop_size) if front_id[i] == f]
                if idx:
                    crowd[idx] = crowding_distance(np.array(idx), F)

            # Selection: tournament on (front_id, -crowd)
            def tournament():
                a, b = rng.integers(pop_size, 2)
                ra = (front_id[a], -crowd[a])
                rb = (front_id[b], -crowd[b])
                return a if ra < rb else b

            # Generate offspring
            mu = pop_size
            Q = np.empty((mu, dim))
            FQ = np.empty((mu, n_obj))
            for q in range(mu):
                if used >= self.budget or time.time() > deadline:
                    break
                p1 = tournament()
                p2 = tournament()
                # Adaptive mutation step
                progress = used / self.budget
                step_size = 0.2 * (1.0 - 0.5 * progress)
                # Blend parents and mutate
                child = 0.5 * (X[p1] + X[p2]) + step_size * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = fit(child)
                used += 1
                Q[q] = child
                FQ[q] = fc

            if used >= self.budget:
                break

            # Combine parent and offspring, select next generation
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, FQ])
            r_front_id, r_n_fronts = fast_non_dominated_sort(R_F)
            r_crowd = np.zeros(2 * pop_size)
            for f in range(r_n_fronts):
                idx = [i for i in range(2 * pop_size) if r_front_id[i] == f]
                if idx:
                    r_crowd[idx] = crowding_distance(np.array(idx), R_F)

            # Sort by (front_id, -crowd) and take top pop_size
            order = np.lexsort((-r_crowd, r_front_id))
            X = R_X[order[:pop_size]]
            F = R_F[order[:pop_size]]

        return X
# EVOLVE-BLOCK-END
