"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(200, max(40, budget // 2))

        # Initial population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def non_dominated_sort(F):
            """Return list of fronts (indices), first front is non-dominated."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination relation
            n_dom = [0] * n
            rank = [0] * n
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            current_front = [i for i in range(n) if n_dom[i] == 0]
            rank_idx = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F, front_indices):
            """Assign crowding distance to indices in front."""
            n = len(front_indices)
            if n <= 2:
                return {idx: float('inf') for idx in front_indices}
            dist = {idx: 0.0 for idx in front_indices}
            for m in range(n_obj):
                vals = sorted(front_indices, key=lambda i: F[i, m])
                dist[vals[0]] = float('inf')
                dist[vals[-1]] = float('inf')
                f_min = F[vals[0], m]
                f_max = F[vals[-1], m]
                if f_max - f_min < 1e-15:
                    continue
                for k in range(1, n - 1):
                    dist[vals[k]] += (F[vals[k + 1], m] - F[vals[k - 1], m]) / (f_max - f_min)
            return dist

        # Main loop
        while used < budget:
            if time.time() - t0 > deadline:
                break

            # Evaluate non-dominated fronts
            fronts = non_dominated_sort(F)

            # Select parents using tournament selection
            parents = []
            for _ in range(pop_size // 2):
                # Tournament of 2
                i, j = rng.integers(0, pop_size, size=2)
                # Determine rank
                rank_i = next(k for k, fr in enumerate(fronts) if i in fr)
                rank_j = next(k for k, fr in enumerate(fronts) if j in fr)
                if rank_i < rank_j:
                    parents.append(i)
                elif rank_j < rank_i:
                    parents.append(j)
                else:
                    # Same front, use crowding distance
                    front_idx = rank_i
                    cd = crowding_distance(F, fronts[front_idx])
                    if cd[i] >= cd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)

            # Generate offspring
            children_X = []
            children_F = []
            n_offspring = pop_size - len(parents)

            for _ in range(n_offspring):
                # Pick two parents
                p1, p2 = parents[rng.integers(0, len(parents), size=2)]
                # Simulated binary crossover
                if rng.random() < 0.9:
                    eta = 20.0
                    u = rng.random(dim)
                    beta1 = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (2 * (1 - u)) ** (1 / (eta + 1)))
                    beta2 = 1.0 / beta1
                    c1 = 0.5 * ((1 + beta1) * X[p1] + (1 - beta1) * X[p2])
                    c2 = 0.5 * ((1 + beta2) * X[p1] + (1 - beta2) * X[p2])
                else:
                    c1 = X[p1].copy()
                    c2 = X[p2].copy()

                # Mutation
                eta_m = 20.0
                for child in [c1, c2]:
                    if rng.random() < 0.9:
                        delta1 = (2 * rng.random(dim)) ** (1 / (eta_m + 1)) - 1
                        delta2 = 1 - (2 * (1 - rng.random(dim))) ** (1 / (eta_m + 1))
                        delta = np.where(rng.random(dim) <= 0.5, delta1, delta2)
                        child += delta * (hi - lo)
                        child = np.clip(child, lo, hi)

                children_X.append(c1)
                children_F.append(problem(c1))
                children_X.append(c2)
                children_F.append(problem(c2))
                used += 2

            if used >= budget:
                break

            # Combine population and offspring
            X_combined = np.vstack([X, np.array(children_X)])
            F_combined = np.vstack([F, np.array(children_F)])

            # Sort into fronts
            fronts = non_dominated_sort(F_combined)

            # Build new population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to select from this front using crowding distance
                    cd = crowding_distance(F_combined, front)
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in front_sorted[:needed]:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
