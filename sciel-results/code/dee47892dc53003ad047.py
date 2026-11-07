"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


class Optimizer:
    """Multi-objective optimizer using NSGA-II with hypervolume-aware selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use a local RNG seeded from the global state for reproducibility
        seed = np.random.get_state()[1][0]
        rng = np.random.RandomState(seed)

        # Population size
        pop_size = min(100, max(20, budget // 4))

        # Initialize with Halton sequence for better coverage
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for d in range(dim):
                X[i, d] = lo[d] + (hi[d] - lo[d]) * self._halton(i + 1, d + 2)

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track all evaluated points for hypervolume (we keep the non-dominated set)
        # But for the algorithm, we maintain a population

        # NSGA-II parameters
        mutation_rate = 0.1
        crossover_rate = 0.9

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            front = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1

                if n_dominates[i] == 0:
                    front[i] = 0
                    fronts.append([i])

            k = 0
            while fronts[k] is not None:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            front[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            return fronts[:k + 1] if k < len(fronts) else fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices in F."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n

            dist = [0.0] * n
            m = F.shape[1]

            for k in range(m):
                obj_vals = [F[idx, k] for idx in indices]
                sorted_idx = sorted(range(n), key=lambda i: obj_vals[i])
                min_val = obj_vals[sorted_idx[0]]
                max_val = obj_vals[sorted_idx[-1]]
                if max_val - min_val < 1e-10:
                    continue
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)

            return dist

        def select_parents(F, X, pop_size):
            """Binary tournament selection based on non-dominated rank and crowding distance."""
            fronts = non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distance per front
            cd = np.zeros(len(F), dtype=np.float64)
            for front in fronts:
                if len(front) > 2:
                    fcd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = fcd[i]

            def compare(i, j):
                if ranks[i] != ranks[j]:
                    return ranks[i] < ranks[j]
                return cd[i] > cd[j]

            # Tournament selection
            parents = []
            for _ in range(pop_size):
                i1 = rng.randint(0, len(F) - 1)
                i2 = rng.randint(0, len(F) - 1)
                if compare(i1, i2):
                    parents.append(i1)
                else:
                    parents.append(i2)
            return parents

        def crossover(x1, x2):
            if rng.rand() > crossover_rate:
                return x1.copy(), x2.copy()
            # Simulated binary crossover (SBX)
            children = [x1.copy(), x2.copy()]
            for d in range(dim):
                if rng.rand() < 0.5:
                    continue
                # SBX
                beta = 20
                x1_val = x1[d]
                x2_val = x2[d]
                if x1_val == x2_val:
                    continue
                # Randomly decide which child gets which value
                if rng.rand() < 0.5:
                    c1, c2 = x1_val, x2_val
                else:
                    c1, c2 = x2_val, x1_val

                # Simplified: use blend crossover for simplicity and speed
                alpha = rng.rand()
                c1 = (1 - alpha) * x1[d] + alpha * x2[d]
                c2 = (1 - alpha) * x2[d] + alpha * x1[d]
                children[0][d] = c1
                children[1][d] = c2
            return children

        def mutate(x):
            # Gaussian mutation
            x = x + mutation_rate * (hi - lo) * rng.randn(dim)
            return np.clip(x, lo, hi)

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Select parents
            parent_indices = select_parents(F, X, pop_size)

            # Create offspring
            offspring_X = np.zeros((pop_size, dim))
            for i in range(pop_size // 2):
                p1_idx = parent_indices[2 * i]
                p2_idx = parent_indices[2 * i + 1]
                c1, c2 = crossover(X[p1_idx], X[p2_idx])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X[2 * i] = c1
                offspring_X[2 * i + 1] = c2

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            if used > budget:
                # Trim if we went over (shouldn't happen with proper sizing)
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            fronts = non_dominated_sort(combined_F)

            # Select next population
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to select from this front
                    remaining = pop_size - count
                    if len(front) > 2:
                        fcd = crowding_distance(combined_F, front)
                        # Sort by crowding distance descending
                        sorted_front = sorted(range(len(front)), key=lambda i: -fcd[i])
                        for i in range(remaining):
                            idx = front[sorted_front[i]]
                            new_X[count] = combined_X[idx]
                            new_F[count] = combined_F[idx]
                            count += 1
                    else:
                        for i in range(remaining):
                            idx = front[i]
                            new_X[count] = combined_X[idx]
                            new_F[count] = combined_F[idx]
                            count += 1
                    break

            X = new_X
            F = new_F

        return X

    def _halton(self, i, base):
        """Generate the i-th Halton sequence value for the given base."""
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
# EVOLVE-BLOCK-END
