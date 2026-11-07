"""Multi-objective optimizer on DTLZ1 using a fast NSGA-II variant with time guard."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size
        pop_size = min(100, max(20, self.budget // 2))
        # For budget=200, pop_size = 100

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (for internal selection if needed)
        # We'll use NSGA-II style selection

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (each is list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = [0] * n  # number of solutions that dominate i
            rank = [0] * n

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1

            # First front
            front = [i for i in range(n) if n_count[i] == 0]
            for i in front:
                rank[i] = 0

            fronts = [front]
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n == 0:
                return []
            if n <= 2:
                return [float('inf')] * n

            m = F.shape[1]
            dist = [0.0] * n

            for k in range(m):
                # Sort by objective k
                sorted_idx = sorted(range(n), key=lambda i: F[indices[i]][k])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_range = F[indices[sorted_idx[-1]]][k] - F[indices[sorted_idx[0]]][k]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F[indices[sorted_idx[i + 1]]][k] - F[indices[sorted_idx[i - 1]]][k]) / f_range

            return dist

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Create offspring
            # Select parents using tournament selection
            offspring_X = []
            offspring_F = []

            # Generate pop_size offspring
            for _ in range(pop_size):
                if time.time() > deadline:
                    break

                # Tournament selection
                p1_idx = self._tournament_select(fronts, F, rng)
                p2_idx = self._tournament_select(fronts, F, rng)

                # Crossover: simulated binary crossover
                parent1 = X[p1_idx]
                parent2 = X[p2_idx]

                # SBX crossover
                eta = 20.0
                child1 = np.empty(self.dim)
                child2 = np.empty(self.dim)
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        if abs(parent1[d] - parent2[d]) < 1e-14:
                            child1[d] = parent1[d]
                            child2[d] = parent2[d]
                        else:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta + 1.0))
                            else:
                                beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                            child1[d] = 0.5 * ((1 + beta) * parent1[d] + (1 - beta) * parent2[d])
                            child2[d] = 0.5 * ((1 - beta) * parent1[d] + (1 + beta) * parent2[d])
                    else:
                        child1[d] = parent2[d]
                        child2[d] = parent1[d]

                # Mutation: polynomial mutation
                eta_m = 20.0
                for d in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        u = rng.random()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                        else:
                            delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                        child1[d] += delta_q * (hi[d] - lo[d])
                    if rng.random() < 1.0 / self.dim:
                        u = rng.random()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                        else:
                            delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                        child2[d] += delta_q * (hi[d] - lo[d])

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                offspring_X.append(child1)
                offspring_F.append(problem(child1))
                used += 1

                if used >= self.budget:
                    break

                if time.time() > deadline:
                    break

                offspring_X.append(child2)
                offspring_F.append(problem(child2))
                used += 1

                if used >= self.budget or time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            n_combined = len(X_combined)

            # Non-dominated sort on combined
            fronts_combined = fast_non_dominated_sort(F_combined)

            # Select new population
            new_X = []
            new_F = []
            i = 0
            while i < len(fronts_combined) and len(new_X) + len(fronts_combined[i]) <= pop_size:
                for idx in fronts_combined[i]:
                    new_X.append(X_combined[idx])
                    new_F.append(F_combined[idx])
                i += 1

            if len(new_X) < pop_size:
                # Need to select from the next front using crowding distance
                remaining = pop_size - len(new_X)
                if i < len(fronts_combined):
                    front_idx = fronts_combined[i]
                    cd = crowding_distance(F_combined, front_idx)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(range(len(front_idx)), key=lambda k: cd[k], reverse=True)
                    for k in sorted_front[:remaining]:
                        idx = front_idx[k]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        return X

    def _tournament_select(self, fronts, F, rng, k=2):
        """Tournament selection based on rank and crowding distance."""
        # Assign ranks
        n = len(F)
        rank = np.zeros(n, dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i

        # Compute crowding distance for each front
        cd = np.zeros(n)
        for front in fronts:
            if len(front) > 2:
                cd_front = crowding_distance(F, front)
                for j, idx in enumerate(front):
                    cd[idx] = cd_front[j]

        # Tournament
        candidates = rng.integers(0, n, size=k)
        best = candidates[0]
        for c in candidates[1:]:
            if rank[c] < rank[best]:
                best = c
            elif rank[c] == rank[best] and cd[c] > cd[best]:
                best = c
        return best
