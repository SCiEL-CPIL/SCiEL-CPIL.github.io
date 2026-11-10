"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with DE/rand/1/bin mutation and crowding-distance-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim

        # Population size
        N = 24
        # DE parameters
        CR = 0.9
        F1 = 0.8

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (each is list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            for i in range(n):
                if n_dom[i] == 0:
                    front0.append(i)

            if front0:
                fronts.append(front0)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1

            return fronts

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_v = vals.min()
                max_v = vals.max()
                if max_v - min_v < 1e-30:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_v - min_v)

            return dist

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = np.zeros((N, dim))
            offspring_F = np.zeros((N, n_obj))

            for i in range(N):
                # Select 3 distinct parents from non-dominated or random
                fronts = fast_non_dominated_sort(F)
                # Prefer parents from first front
                if fronts and len(fronts[0]) >= 3:
                    parent_pool = fronts[0]
                elif fronts:
                    parent_pool = list(range(N))
                else:
                    parent_pool = list(range(N))

                idxs = rng.choice(len(parent_pool), size=3, replace=False)
                p1 = X[parent_pool[idxs[0]]]
                p2 = X[parent_pool[idxs[1]]]
                p3 = X[parent_pool[idxs[2]]]

                # DE/rand/1/bin
                v = p1 + F1 * (p2 - p3)
                # Binary mask
                mask = rng.random(dim) < CR
                if not np.any(mask):
                    mask[rng.integers(dim)] = True
                child = np.where(mask, v, p1)
                child = np.clip(child, lo, hi)

                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

                # Check deadline inside inner loop too
                if time.time() > deadline:
                    # Fill remaining with copies
                    for j in range(i + 1, N):
                        offspring_X[j] = child
                        offspring_F[j] = offspring_F[i]
                    break

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort
            fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = np.zeros((N, dim))
            next_F = np.zeros((N, n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= N:
                    for idx in front:
                        next_X[count] = combined_X[idx]
                        next_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to select from this front using crowding distance
                    remaining = N - count
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(cd)[::-1]
                    for k in range(remaining):
                        idx = front[order[k]]
                        next_X[count] = combined_X[idx]
                        next_F[count] = combined_F[idx]
                        count += 1
                    break

            X = next_X
            F = next_F

        return X
# EVOLVE-BLOCK-END
