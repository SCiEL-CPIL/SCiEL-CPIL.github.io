"""Multi-objective optimizer for RE37: NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with deadline safety."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 30

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        pop_size = self.pop_size

        # Use a fixed seed for reproducibility within the optimizer
        # The evaluator calls np.random.seed(seed) before creating Optimizer
        # But we should use our own rng to not interfere with global state
        # Actually, the evaluator calls np.random.seed(seed), so global state is set.
        # Let's use np.random directly or a seeded rng.
        # The task says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)"
        # This means the global numpy random state is seeded.
        # We can use np.random directly, or create a default_rng.
        # Using default_rng(0) would be deterministic regardless of evaluator seed.
        # But the evaluator expects different results for different seeds.
        # So we should use the global np.random state which is seeded by the evaluator.
        # Let's use np.random.

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, d))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = np.zeros(n, dtype=int)  # n_d[i] = number of points dominating i
            rank = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1
                if n_d[i] == 0:
                    rank[i] = 0
                    fronts.append([i])

            i = 0
            while i < len(fronts):
                current_front = fronts[i]
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                i += 1

            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.ones(len(indices)) * np.inf

            m = self.n_obj
            n = len(indices)
            dist = np.zeros(n)

            for k in range(m):
                # Sort by k-th objective
                sorted_idx = np.argsort(F[indices, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F[indices[sorted_idx[-1]], k] - F[indices[sorted_idx[0]], k]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F[indices[sorted_idx[i + 1]], k] - F[indices[sorted_idx[i - 1]], k]) / f_range

            return dist

        def tournament_selection(fronts, F, X, k=2):
            """Tournament selection based on rank and crowding distance."""
            # Pre-compute ranks
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i

            # Pre-compute crowding distances
            cd = np.zeros(n)
            for front in fronts:
                if len(front) > 2:
                    cd[front] = crowding_distance(F, front)
                else:
                    cd[front] = np.inf

            selected = np.zeros(d)
            for _ in range(2):
                idx1 = np.random.randint(n)
                idx2 = np.random.randint(n)
                if ranks[idx1] < ranks[idx2]:
                    winner = idx1
                elif ranks[idx2] < ranks[idx1]:
                    winner = idx2
                else:
                    winner = idx1 if cd[idx1] > cd[idx2] else idx2
                selected += X[winner]
            return selected / 2

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Evaluate if we have time for a full generation
            # Each generation uses pop_size evaluations
            # We'll do partial generations if needed

            # Create fronts
            fronts = fast_non_dominated_sort(F)

            # Select parents and create offspring
            offspring_X = np.zeros((pop_size, d))
            offspring_F = np.zeros((pop_size, self.n_obj))

            for i in range(pop_size):
                # Check deadline before each evaluation
                if time.time() > deadline:
                    break

                # Tournament selection
                parent = tournament_selection(fronts, F, X)

                # Mutation: Gaussian mutation
                sigma = 0.1 * (hi - lo)
                child = parent + sigma * np.random.standard_normal(d)
                child = np.clip(child, lo, hi)

                # Evaluate
                offspring_F[i] = problem(child)
                offspring_X[i] = child
                used += 1

            if used >= self.budget:
                break

            # Combine parents and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            fronts_combined = fast_non_dominated_sort(F_combined)

            # Select next generation
            new_X = np.zeros((pop_size, d))
            new_F = np.zeros((pop_size, self.n_obj))
            count = 0

            for front in fronts_combined:
                if count + len(front) <= pop_size:
                    # Take all from this front
                    for idx in front:
                        new_X[count] = X_combined[idx]
                        new_F[count] = F_combined[idx]
                        count += 1
                else:
                    # Need to select partially from this front
                    # Calculate crowding distance
                    cd = crowding_distance(F_combined, front)
                    sorted_by_cd = np.argsort(-cd)
                    needed = pop_size - count
                    for j in range(needed):
                        idx = front[sorted_by_cd[j]]
                        new_X[count] = X_combined[idx]
                        new_F[count] = F_combined[idx]
                        count += 1
                    break

            X = new_X
            F = new_F

        return X
# EVOLVE-BLOCK-END
