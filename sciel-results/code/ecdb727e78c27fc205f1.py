# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a good chunk of budget
        pop_size = int(min(50, max(20, budget // 4)))

        def dominates(a, b):
            """Return True if a dominates b."""
            return np.all(a <= b) and np.any(a < b)

        def fast_nondominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (each front is a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            nd = np.zeros(n, dtype=int)  # nd[i] = number of solutions dominating i
            ranks = np.full(n, -1, dtype=int)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    ranks[i] = 0

            fronts = []
            current = [i for i in range(n) if nd[i] == 0]
            if current:
                fronts.append(current)

            k = 0
            while current:
                next_front = []
                for i in current:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                current = next_front

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)

            cd = np.zeros(m)
            idx = np.array(indices)
            subF = F[idx]

            for o in range(n_obj):
                sorted_idx = np.argsort(subF[:, o])
                min_v = subF[sorted_idx[0], o]
                max_v = subF[sorted_idx[-1], o]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                rng_v = max_v - min_v
                if rng_v < 1e-12:
                    continue
                for kk in range(1, m - 1):
                    cd[sorted_idx[kk]] += (subF[sorted_idx[kk + 1], o] - subF[sorted_idx[kk - 1], o]) / rng_v

            return cd

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track all non-dominated solutions found so far
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            """Add to archive if not dominated, remove dominated points."""
            nonlocal archive_X, archive_F
            # Check if f is dominated by any point in archive
            for g in archive_F:
                if dominates(g, f):
                    return
            # Remove points dominated by f
            new_X = []
            new_F = []
            for i in range(len(archive_X)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F

        # Initialize archive
        for i in range(pop_size):
            update_archive(X[i], F[i])

        while used < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Get current non-dominated front
            fronts = fast_nondominated_sort(F)
            if not fronts:
                break
            front0 = fronts[0]
            if not front0:
                front0 = list(range(pop_size))

            # Binary tournament selection from front0
            i1 = front0[rng.integers(len(front0))]
            i2 = front0[rng.integers(len(front0))]
            if dominates(F[i1], F[i2]):
                parent_idx = i1
            else:
                parent_idx = i2

            # Adaptive mutation: larger steps early, smaller later
            progress = used / budget
            mutation_scale = 0.3 * (1.0 - progress) + 0.05
            mutation = mutation_scale * (hi - lo) * rng.standard_normal(dim)
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = np.array(problem(child), dtype=np.float64)
            used += 1

            # Update archive
            update_archive(child, fc)

            # Replace worst point in population (by sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Occasionally replace a random point for diversity
            if rng.random() < 0.2:
                rand_idx = rng.integers(pop_size)
                X[rand_idx] = child
                F[rand_idx] = fc

        return X
# EVOLVE-BLOCK-END
