"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with DE-based offspring and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sorting (Deb et al. 2002). Returns list of fronts (lists of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # dominance sets
        n_dom = np.zeros(n, dtype=int)  # number of points dominated by i
        rank = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        for i in current_front:
            rank[i] = 0

        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            current_front = next_front
            k += 1

        return fronts, rank

    @staticmethod
    def _crowding_distance(F, front_indices):
        """Crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        m = F.shape[1] if hasattr(F, 'shape') else len(F[0])

        for obj in range(m):
            sorted_idx = np.argsort([F[i, obj] for i in front_indices])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            f_min = F[front_indices[sorted_idx[0]], obj]
            f_max = F[front_indices[sorted_idx[-1]], obj]
            if f_max - f_min < 1e-15:
                continue
            for k in range(1, n - 1):
                i1 = front_indices[sorted_idx[k - 1]]
                i2 = front_indices[sorted_idx[k + 1]]
                dist[sorted_idx[k]] += (F[i2, obj] - F[i1, obj]) / (f_max - f_min)

        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = 40
        # DE parameters
        CR = 0.8
        F_DE = 0.5

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts, rank = self._fast_non_dominated_sort(F)

            # Build offspring
            offspring = []
            while len(offspring) < pop_size and used < self.budget:
                if time.time() > deadline:
                    break

                # Select parents via tournament on rank + crowding
                # Compute crowding distance for each front
                crowding = np.zeros(pop_size)
                for front in fronts:
                    cd = self._crowding_distance(F, front)
                    for idx, fi in enumerate(front):
                        crowding[fi] = cd[idx]

                # Tournament selection
                def tournament():
                    a = rng.integers(pop_size)
                    b = rng.integers(pop_size)
                    if rank[a] < rank[b]:
                        return a
                    elif rank[b] < rank[a]:
                        return b
                    else:
                        return a if crowding[a] > crowding[b] else b

                # DE-based offspring: pick 3 distinct parents
                p1, p2, p3 = tournament(), tournament(), tournament()
                while p2 == p1:
                    p2 = tournament()
                while p3 == p1 or p3 == p2:
                    p3 = tournament()

                v = X[p1] + F_DE * (X[p2] - X[p3])
                # Crossover
                trial = np.empty(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == rng.integers(self.dim):
                        trial[j] = v[j]
                    else:
                        trial[j] = X[p1][j]

                trial = np.clip(trial, lo, hi)
                ft = problem(trial)
                used += 1
                offspring.append((trial, ft))

            if not offspring:
                break

            # Combine parent and offspring
            X_comb = np.vstack([X, np.array([o[0] for o in offspring])])
            F_comb = np.array([F, np.array([o[1] for o in offspring])])

            # Selection: keep best pop_size from combined
            fronts_comb, rank_comb = self._fast_non_dominated_sort(F_comb)
            crowding_comb = np.zeros(len(F_comb))
            for front in fronts_comb:
                cd = self._crowding_distance(F_comb, front)
                for idx, fi in enumerate(front):
                    crowding_comb[fi] = cd[idx]

            # Sort by rank then crowding (descending)
            order = np.argsort(rank_comb - crowding_comb / (1 + np.max(crowding_comb)))
            new_indices = order[:pop_size]

            X = X_comb[new_indices]
            F = F_comb[new_indices]

            # Record
            all_X.append(X.copy())
            all_F.append(F.copy())

        return X
# EVOLVE-BLOCK-END
