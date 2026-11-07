"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style selection with archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 280  # Safety margin (limit is 300s)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        N = 20
        if self.budget < N * 2:
            N = max(2, self.budget // 2)

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        evals = N

        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = F.shape[0]
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of solutions that dominate i

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            current_front = [i for i in range(n) if n_dom[i] == 0]
            rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = rank
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                rank += 1
                current_front = next_front
            return ranks

        # Helper: Crowding distance
        def crowding_distance(F, ranks):
            n = F.shape[0]
            distances = np.zeros(n)
            if n <= 2:
                return distances

            # Group by rank
            max_rank = np.max(ranks)
            for r in range(max_rank + 1):
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    continue

                for m in range(self.n_obj):
                    # Sort by m-th objective
                    sorted_idx = indices[np.argsort(F[indices, m])]
                    distances[sorted_idx[0]] = np.inf
                    distances[sorted_idx[-1]] = np.inf

                    obj_min = F[sorted_idx[0], m]
                    obj_max = F[sorted_idx[-1], m]
                    obj_range = obj_max - obj_min
                    if obj_range == 0:
                        continue

                    for i in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[i]
                        prev_idx = sorted_idx[i - 1]
                        next_idx = sorted_idx[i + 1]
                        distances[idx] += (F[next_idx, m] - F[prev_idx, m]) / obj_range

            return distances

        # Main Loop
        while evals < self.budget:
            if time.time() - start_time > timeout:
                break

            # Select parents using tournament selection on (rank, crowding distance)
            ranks = non_dominated_sort(F)
            cd = crowding_distance(F, ranks)

            # Select N parents
            parents_idx = []
            for _ in range(N):
                # Tournament size 2
                i, j = rng.integers(0, N, size=2)
                # Compare rank first
                if ranks[i] != ranks[j]:
                    p = i if ranks[i] < ranks[j] else j
                else:
                    # Compare crowding distance
                    p = i if cd[i] > cd[j] else j
                parents_idx.append(p)

            # Generate offspring
            offspring_X = []
            for p in parents_idx:
                # Gaussian Mutation
                # Sigma proportional to range
                sigma = 0.2 * (hi - lo)
                child = X[p] + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            offspring_X = np.array(offspring_X)

            # Evaluate offspring
            # Check time again
            if time.time() - start_time > timeout:
                break

            offspring_F = np.array([problem(x) for x in offspring_X])
            evals += N

            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])

            # Sort and Select
            ranks_comb = non_dominated_sort(F_comb)
            cd_comb = crowding_distance(F_comb, ranks_comb)

            # Create selection key: (rank, -crowding_distance)
            # We want low rank, high crowding distance
            # Sort indices
            selection_keys = np.stack([ranks_comb, -cd_comb], axis=1)
            # argsort is stable? No, but we just need top N.
            # Let's use a custom sort or just iterate.
            # Since 2N is small, we can just sort.
            order = np.lexsort((selection_keys[:, 1], selection_keys[:, 0]))

            X = X_comb[order[:N]]
            F = F_comb[order[:N]]

        return X
# EVOLVE-BLOCK-END
