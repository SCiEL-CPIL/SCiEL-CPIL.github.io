"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """NSGA-II style multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the global numpy random state (set by evaluator with np.random.seed)
        # But we also create our own for reproducibility within the algorithm
        rng = np.random.default_rng()

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all objectives and < in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """
            Fast non-dominated sorting.
            Returns list of fronts, each front is a list of indices.
            Also assigns crowding distances.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_d = [0] * n  # number of points that dominate i
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1
                if n_d[i] == 0:
                    fronts[0].append(i)

            i = 0
            while len(fronts[i]) > 0:
                next_front = []
                for k in fronts[i]:
                    for m in S[k]:
                        n_d[m] -= 1
                        if n_d[m] == 0:
                            next_front.append(m)
                i += 1
                if next_front:
                    fronts.append(next_front)

            # Remove empty last front if any
            if len(fronts) > 1 and len(fronts[-1]) == 0:
                fronts.pop()

            # Assign crowding distances
            crowding = np.zeros(n)
            for front in fronts:
                if len(front) <= 2:
                    for idx in front:
                        crowding[idx] = float('inf')
                    continue
                for m in range(n_obj):
                    front_sorted = sorted(front, key=lambda idx: F[idx, m])
                    crowding[front_sorted[0]] = float('inf')
                    crowding[front_sorted[-1]] = float('inf')
                    f_min = F[front_sorted[0], m]
                    f_max = F[front_sorted[-1], m]
                    if f_max - f_min == 0:
                        continue
                    for k in range(1, len(front_sorted) - 1):
                        idx = front_sorted[k]
                        idx_next = front_sorted[k + 1]
                        idx_prev = front_sorted[k - 1]
                        crowding[idx] += (F[idx_next, m] - F[idx_prev, m]) / (f_max - f_min)

            return fronts, crowding

        def tournament_selection(F, crowding, size=2):
            """Tournament selection based on non-dominated rank and crowding distance."""
            n = len(F)
            # Compute rank for each point
            fronts, _ = fast_non_dominated_sort(F)
            rank = np.zeros(n, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            idxs = rng.integers(n, size=size)
            best = idxs[0]
            for idx in idxs[1:]:
                if rank[idx] < rank[best] or (rank[idx] == rank[best] and crowding[idx] > crowding[best]):
                    best = idx
            return best

        def crossover(x1, x2):
            """Simulated binary crossover."""
            eta = 20.0
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        def mutation(x, sigma):
            """Gaussian mutation."""
            return x + sigma * rng.standard_normal(dim)

        # Initialize population
        pop_size = min(50, budget // 4)
        pop_size = max(pop_size, 10)

        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Initial non-dominated sort and crowding
        fronts, crowding = fast_non_dominated_sort(F)

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Create next generation
            offspring_X = []
            offspring_F = []

            # Generate offspring until we have pop_size of them
            while len(offspring_X) < pop_size and used < budget:
                if time.time() > deadline:
                    break

                # Selection
                idx1 = tournament_selection(F, crowding)
                idx2 = tournament_selection(F, crowding)

                if idx1 == idx2:
                    # Clone
                    child = X[idx1].copy()
                else:
                    # Crossover
                    c1, c2 = crossover(X[idx1], X[idx2])
                    # Mutate both
                    sigma = 0.1 * (hi - lo)
                    c1 = mutation(c1, sigma)
                    c2 = mutation(c2, sigma)
                    c1 = np.clip(c1, lo, hi)
                    c2 = np.clip(c2, lo, hi)

                    # Evaluate first child
                    f1 = problem(c1)
                    used += 1
                    offspring_X.append(c1)
                    offspring_F.append(f1)

                    if used >= budget:
                        break

                    # Evaluate second child
                    f2 = problem(c2)
                    used += 1
                    offspring_X.append(c2)
                    offspring_F.append(f2)
                continue

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, offspring_F])

            # Non-dominated sort on combined population
            fronts_comb, crowding_comb = fast_non_dominated_sort(F_comb)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_comb:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    front_sorted = sorted(front, key=lambda idx: -crowding_comb[idx])
                    for idx in front_sorted[:remaining]:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

            # Recompute fronts and crowding for next iteration
            fronts, crowding = fast_non_dominated_sort(F)

        return X
# EVOLVE-BLOCK-END
