# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
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

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        return fronts

    def _crowding_distance(self, F, indices):
        if len(indices) <= 2:
            return {i: float('inf') for i in indices}
        m = len(F[0])
        dist = {i: 0.0 for i in indices}
        for j in range(m):
            sorted_idx = sorted(indices, key=lambda i: F[i][j])
            dist[sorted_idx[0]] = float('inf')
            dist[sorted_idx[-1]] = float('inf')
            f_min = F[sorted_idx[0]][j]
            f_max = F[sorted_idx[-1]][j]
            if f_max == f_min:
                continue
            for k in range(1, len(sorted_idx) - 1):
                dist[sorted_idx[k]] += (F[sorted_idx[k + 1]][j] - F[sorted_idx[k - 1]][j]) / (f_max - f_min)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use a fixed seed for reproducibility within the call
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 20
        if budget < pop_size:
            pop_size = max(2, budget)

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals_used = pop_size

        # Reference point for hypervolume calculation (for internal selection if needed)
        # But we'll use standard NSGA-II selection

        while evals_used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Build selection pool: prioritize lower fronts
            # We need to select parents. Let's do tournament selection based on rank and crowding.

            # Assign ranks
            rank = [0] * len(F)
            for front_idx, front in enumerate(fronts):
                for i in front:
                    rank[i] = front_idx

            # Compute crowding distance for each front
            crowd = [0.0] * len(F)
            for front in fronts:
                if len(front) <= 2:
                    for i in front:
                        crowd[i] = float('inf')
                else:
                    cd = self._crowding_distance(F, front)
                    for i in front:
                        crowd[i] = cd[i]

            # Selection: binary tournament
            def tournament():
                i1 = rng.integers(len(F))
                i2 = rng.integers(len(F))
                if rank[i1] < rank[i2]:
                    return i1
                elif rank[i2] < rank[i1]:
                    return i2
                else:
                    if crowd[i1] > crowd[i2]:
                        return i1
                    else:
                        return i2

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size

            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                p1 = tournament()
                p2 = tournament()
                # Simulated binary crossover (SBX)
                child1 = X[p1].copy()
                child2 = X[p2].copy()
                eta = 15.0
                for j in range(dim):
                    if rng.random() < 0.5:
                        # Crossover
                        if abs(X[p1][j] - X[p2][j]) > 1e-10:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta + 1.0))
                            else:
                                beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                            child1[j] = 0.5 * ((1 + beta) * X[p1][j] + (1 - beta) * X[p2][j])
                            child2[j] = 0.5 * ((1 - beta) * X[p1][j] + (1 + beta) * X[p2][j])
                        else:
                            child1[j] = X[p1][j]
                            child2[j] = X[p2][j]

                # Polynomial mutation
                for child in [child1, child2]:
                    for j in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                            else:
                                delta_q = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                            child[j] += delta_q * (hi[j] - lo[j])

                # Clip to bounds
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                f1 = problem(child1)
                f2 = problem(child2)
                evals_used += 2
                offspring_X.append(child1)
                offspring_F.append(f1)
                offspring_X.append(child2)
                offspring_F.append(f2)

                if evals_used >= budget:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)
            offspring_X = np.array(offspring_X)

            # Combine parents and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            fronts_all = self._fast_non_dominated_sort(F_all)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts_all:
                if len(next_X) + len(front) <= pop_size:
                    for i in front:
                        next_X.append(X_all[i])
                        next_F.append(F_all[i])
                else:
                    # Need to truncate this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    # Compute crowding distance for this front
                    cd = self._crowding_distance(F_all, front)
                    # Sort by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    for i in front_sorted[:remaining]:
                        next_X.append(X_all[i])
                        next_F.append(F_all[i])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
