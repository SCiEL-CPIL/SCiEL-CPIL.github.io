"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        pop_size = 40
        # Ensure we don't exceed budget
        if pop_size > self.budget:
            pop_size = max(10, self.budget // 2)

        # Initialize population
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """NSGA-II fast non-dominated sort. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = [[]]

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    fronts[0].append(p)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)

            return fronts[:-1]

        def crowding_distance(F, front):
            """Compute crowding distance for a front."""
            n = len(front)
            if n <= 2:
                return [float('inf')] * n
            cd = np.zeros(n)
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F[front, m])
                cd_sorted = np.zeros(n)
                cd_sorted[0] = float('inf')
                cd_sorted[-1] = float('inf')
                for i in range(1, n - 1):
                    denom = F[front[sorted_idx[i + 1]], m] - F[front[sorted_idx[i - 1]], m]
                    if denom > 1e-14:
                        cd_sorted[i] = (F[front[sorted_idx[i + 1]], m] - F[front[sorted_idx[i - 1]], m]) / denom
                    else:
                        cd_sorted[i] = 0
                # Map back
                for i, idx in enumerate(sorted_idx):
                    cd[np.where(front == idx)[0][0]] = cd_sorted[i]
            return cd

        # Main evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Create offspring
            offspring_X = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Select parents using tournament selection based on fronts and crowding
                # Assign ranks
                rank = np.zeros(len(F), dtype=int)
                for fi, front in enumerate(fronts):
                    for idx in front:
                        rank[idx] = fi

                # Compute crowding distances per front
                cd = np.zeros(len(F))
                for front in fronts:
                    if len(front) > 0:
                        front_cd = crowding_distance(F, np.array(front))
                        for j, idx in enumerate(front):
                            cd[idx] = front_cd[j]

                # Tournament selection
                parent1_idx = self._tournament_select(rank, cd, 2)
                parent2_idx = self._tournament_select(rank, cd, 2)

                # Crossover (SBX-like via uniform crossover with mutation)
                if np.random.random() < 0.9:
                    # Blend crossover
                    alpha = np.random.random(self.dim)
                    child = alpha * X[parent1_idx] + (1 - alpha) * X[parent2_idx]
                else:
                    child = X[parent1_idx].copy()

                # Mutation
                if np.random.random() < 0.9:
                    # Polynomial-like mutation
                    mutation_mask = np.random.random(self.dim) < 0.3
                    if np.any(mutation_mask):
                        child[mutation_mask] += 0.1 * span[mutation_mask] * np.random.standard_normal(np.sum(mutation_mask))

                child = np.clip(child, lo, hi)
                offspring_X[i] = child

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Select next generation
            fronts = fast_non_dominated_sort(combined_F)

            next_X = np.zeros((pop_size, self.dim))
            next_F = np.zeros((pop_size, self.n_obj))
            count = 0

            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        next_X[count] = combined_X[idx]
                        next_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to select some from this front based on crowding distance
                    front_arr = np.array(front)
                    front_cd = crowding_distance(combined_F, front_arr)
                    # Sort by crowding distance descending
                    sorted_idx = np.argsort(front_cd)[::-1]
                    needed = pop_size - count
                    for j in range(min(needed, len(front_arr))):
                        idx = front_arr[sorted_idx[j]]
                        next_X[count] = combined_X[idx]
                        next_F[count] = combined_F[idx]
                        count += 1
                    break

            X = next_X[:count]
            F = next_F[:count]

        return X

    def _tournament_select(self, rank, cd, k):
        """Tournament selection based on rank and crowding distance."""
        candidates = np.random.choice(len(rank), size=k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if rank[c] < rank[best] or (rank[c] == rank[best] and cd[c] > cd[best]):
                best = c
        return best
# EVOLVE-BLOCK-END
