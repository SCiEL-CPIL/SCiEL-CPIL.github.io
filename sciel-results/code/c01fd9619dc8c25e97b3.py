"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective search with NSGA-II style selection and CMA-ES-like mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Population size: use a reasonable fraction of budget
        # For 200 budget, use pop_size around 40-50
        pop_size = min(40, max(10, self.budget // 4))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: fast non-dominated sorting (simplified)
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]  # domination count
            n_dom = [0] * n  # number dominated
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1

            for i in range(n):
                if n_dom[i] == 0:
                    front0.append(i)
            fronts.append(front0)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts

        # Helper: crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            m = len(F)
            cd = np.zeros(m)
            for i in indices:
                for d in range(self.n_obj):
                    vals = [F[j][d] for j in indices]
                    min_val = min(vals)
                    max_val = max(vals)
                    if max_val - min_val < 1e-15:
                        cd[i] += float('inf')
                    else:
                        # Find neighbors
                        sorted_idx = sorted(indices, key=lambda j: F[j][d])
                        pos = sorted_idx.index(i)
                        if pos == 0 or pos == len(sorted_idx) - 1:
                            cd[i] += float('inf')
                        else:
                            prev_j = sorted_idx[pos - 1]
                            next_j = sorted_idx[pos + 1]
                            cd[i] += (F[next_j][d] - F[prev_j][d]) / (max_val - min_val)
            return cd

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Select parents from first front(s)
            parent_pool = []
            for front in fronts:
                if len(parent_pool) + len(front) <= pop_size // 2:
                    parent_pool.extend(front)
                else:
                    # Use crowding distance to select remaining
                    remaining = pop_size // 2 - len(parent_pool)
                    cd = crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    parent_pool.extend(sorted_front[:remaining])
                    break

            if not parent_pool:
                parent_pool = list(range(pop_size))

            # Generate offspring
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []
            for _ in range(n_offspring):
                # Select two parents via tournament
                p1_idx = parent_pool[rng.integers(len(parent_pool))]
                p2_idx = parent_pool[rng.integers(len(parent_pool))]

                # SBX-like crossover
                if rng.random() < 0.8:
                    beta = 15.0
                    u = rng.random()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta_q = (alpha + 1.0) ** (-1.0 / (beta + 1.0))
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta_q = (alpha + 1.0) ** (-1.0 / (beta + 1.0))

                    x1 = X[p1_idx]
                    x2 = X[p2_idx]
                    child1 = 0.5 * ((1.0 + beta_q) * x1 + (1.0 - beta_q) * x2)
                    child2 = 0.5 * ((1.0 + 1.0 / beta_q) * x1 + (1.0 - 1.0 / beta_q) * x2)

                    # Mutate with polynomial mutation
                    eta_m = 20.0
                    mask = rng.random(self.dim) < 0.1
                    if np.any(mask):
                        u = rng.random(self.dim)
                        u[mask] = u[mask]
                        for d in range(self.dim):
                            if mask[d]:
                                if u[d] < 0.5:
                                    delta_q = (2.0 * u[d]) ** (1.0 / (eta_m + 1.0)) - 1.0
                                else:
                                    delta_q = 1.0 - (2.0 * (1.0 - u[d])) ** (1.0 / (eta_m + 1.0))
                                child1[d] += delta_q * (hi[d] - lo[d])
                                child2[d] -= delta_q * (hi[d] - lo[d])

                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                else:
                    child1 = X[p1_idx].copy()
                    child2 = X[p2_idx].copy()
                    # Mutate
                    mask = rng.random(self.dim) < 0.1
                    if np.any(mask):
                        u = rng.random(self.dim)
                        for d in range(self.dim):
                            if mask[d]:
                                if u[d] < 0.5:
                                    delta_q = (2.0 * u[d]) ** (1.0 / (21.0)) - 1.0
                                else:
                                    delta_q = 1.0 - (2.0 * (1.0 - u[d])) ** (1.0 / (21.0))
                                child1[d] += delta_q * (hi[d] - lo[d])
                                child2[d] += delta_q * (hi[d] - lo[d])
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)

                offspring_X.append(child1)
                offspring_F.append(problem(child1))
                used += 1
                if time.time() > deadline:
                    break

                if used < self.budget:
                    offspring_X.append(child2)
                    offspring_F.append(problem(child2))
                    used += 1

            if not offspring_X:
                break

            # Combine parent and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])

            # Non-dominated sort combined
            fronts = non_dominated_sort(F_combined)

            # Select next population
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    remaining = pop_size - len(new_indices)
                    cd = crowding_distance(F_combined, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    new_indices.extend(sorted_front[:remaining])
                    break

            if len(new_indices) < pop_size:
                # Fill with best from last front
                new_indices.extend(range(len(F_combined)))

            new_indices = new_indices[:pop_size]
            X = X_combined[new_indices]
            F = F_combined[new_indices]

        return X
# EVOLVE-BLOCK-END
