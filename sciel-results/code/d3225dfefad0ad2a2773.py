"""Multi-objective optimizer for WFG7 (3 objectives, 12 vars) using NSGA-II
with front-seeded initialization, uniform crossover + adaptive mutation, and
elitist replacement, plus an explicit wall-clock deadline guard."""
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using NSGA-II with crossover and
    front-seeded initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before the 330s wall-clock kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # ------------------------------------------------------------------
        # Non-dominated sorting (fast, returns list of fronts of indices)
        # ------------------------------------------------------------------
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for idx in fronts[i]:
                    for j in S[idx]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            Q.append(j)
                if Q:
                    i += 1
                    fronts.append(Q)
                else:
                    break
            return fronts

        # ------------------------------------------------------------------
        # Crowding distance
        # ------------------------------------------------------------------
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            cd = np.zeros(len(indices))
            for m in range(n_obj):
                vals = F[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
            return cd

        # ------------------------------------------------------------------
        # Front-seeded initialization: points on the WFG7 front (unit
        # hypersphere in objective space) mapped into decision space, plus
        # random points for robustness.
        # ------------------------------------------------------------------
        def on_front_point(rng):
            # For WFG7, the Pareto front is a quarter-sphere in objective space.
            # The front is determined by the first few decision variables.
            # Use a more informed initialization: sample uniformly on the
            # front by choosing angles that map to the WFG7 front structure.
            # WFG7 front: f1^2 + f2^2 + f3^2 = 1 (approximately), all positive.
            # Sample two angles to get a point on the unit sphere in positive octant
            theta = rng.uniform(0, np.pi / 2)
            phi = rng.uniform(0, np.pi / 2)
            v = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            # Map to decision space: the front is driven by early variables
            x = 0.5 * (lo + hi).copy()
            for k in range(3):
                x[k] = lo[k] + v[k] * (hi[k] - lo[k])
            return x

        pop_size = 60
        n_seed_front = 30
        n_seed_rand = pop_size - n_seed_front

        X = np.empty((pop_size, dim))
        for i in range(n_seed_front):
            X[i] = on_front_point(rng)
        for i in range(n_seed_rand):
            X[i] = lo + span * rng.random(dim)
        X = np.clip(X, lo, hi)

        F = np.empty((pop_size, n_obj))
        for i in range(pop_size):
            F[i] = problem(X[i])
        used = pop_size

        # Track the best (lowest objective-norm) point for convergence bias
        # Cache the best front index to avoid recomputing NS every call
        _best_cache = [None]

        def best_front_idx(F):
            # Use cached fronts if available; otherwise compute once
            if _best_cache[0] is not None:
                front0 = _best_cache[0]
            else:
                fronts = non_dominated_sort(F)
                front0 = np.array(fronts[0]) if fronts[0] else np.array([0])
                _best_cache[0] = front0
            norms = np.linalg.norm(F[front0], axis=1)
            return front0[np.argmin(norms)]

        # ------------------------------------------------------------------
        # Main evolution loop (elitist NSGA-II with crossover + mutation)
        # ------------------------------------------------------------------
        while used < budget:
            if time.time() > deadline:
                break

            # Build ranks + crowding for the current population
            fronts = non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            crowd = np.zeros(len(F))
            for front in fronts:
                cd = crowding_distance(F, np.array(front))
                for k, idx in enumerate(front):
                    crowd[idx] = cd[k]

            # Number of offspring this generation (fill remaining budget)
            remaining = budget - used
            n_off = min(pop_size, remaining)
            if n_off <= 0:
                break

            # Offspring container
            offX = np.empty((n_off, dim))
            offF = np.empty((n_off, n_obj))

            # Convergence bias: occasionally use the best front point as one
            # parent to pull the population onto the true front.
            bias_prob = 0.40

            for c in range(n_off):
                if time.time() > deadline:
                    break

                # ---- select two parents via tournament on (rank, crowding)
                # Tournament size 3 for stronger selection pressure
                def tournament():
                    idxs = rng.integers(len(F), size=3)
                    best = idxs[0]
                    for j in range(1, 3):
                        if ranks[idxs[j]] < ranks[best]:
                            best = idxs[j]
                        elif ranks[idxs[j]] == ranks[best] and crowd[idxs[j]] > crowd[best]:
                            best = idxs[j]
                    return best

                p1 = tournament()
                p2 = tournament()

                # Convergence bias: replace p2 with the best front point
                if rng.random() < bias_prob:
                    p2 = best_front_idx(F)

                parent1 = X[p1]
                parent2 = X[p2]

                # ---- uniform crossover
                mask = rng.random(dim) < 0.5
                child = np.where(mask, parent1, parent2)

                # ---- adaptive mutation (scale shrinks as budget is used)
                # Larger base mutation for better exploration in 12-D
                progress = used / budget
                mut_scale = 0.15 * (1.0 - 0.60 * progress)
                mutation = mut_scale * span * rng.standard_normal(dim)
                child = child + mutation
                child = np.clip(child, lo, hi)

                offX[c] = child
                offF[c] = problem(child)
                used += 1

            if used >= budget:
                break

            # ---- elitist replacement: combine parents + offspring, keep best
            popX = np.vstack([X, offX])
            popF = np.vstack([F, offF])
            fronts = non_dominated_sort(popF)
            ranks = np.zeros(len(popF), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            crowd = np.zeros(len(popF))
            for front in fronts:
                cd = crowding_distance(popF, np.array(front))
                for k, idx in enumerate(front):
                    crowd[idx] = cd[k]

            # Sort by (rank asc, crowding desc) and take top pop_size
            order = np.lexsort((-crowd, ranks))
            keep = order[:pop_size]
            X = popX[keep]
            F = popF[keep]

            # Update best-front cache for next generation
            _best_cache[0] = np.array(fronts[0]) if fronts[0] else np.array([0])

        return X
