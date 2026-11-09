import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = min(50, budget // 4)
        N = max(N, 10)

        # Initial population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each front is a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_d = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.ones(len(indices)) * np.inf

            dist = np.zeros(len(indices))
            F_sub = F[indices]

            for m in range(n_obj):
                order = np.argsort(F_sub[:, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = F_sub[order[-1], m] - F_sub[order[0], m]
                if f_range > 0:
                    for k in range(1, len(order) - 1):
                        dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / f_range

            return dist

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sort into fronts
            fronts = fast_non_dominated_sort(F)

            # Create mating pool from first few fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) >= N:
                    break
                mating_pool.extend(front)

            if not mating_pool:
                mating_pool = list(range(N))

            # Generate offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < N:
                if time.time() > deadline:
                    break

                # Binary tournament
                p1, p2 = rng.integers(0, len(mating_pool), size=2)
                parent_idx = mating_pool[p1] if p1 < p2 else mating_pool[p2]
                # Actually, let's do proper tournament selection based on front rank
                # For simplicity, pick from mating_pool randomly
                parent_idx = mating_pool[rng.integers(len(mating_pool))]

                # SBX crossover
                if rng.random() < 0.9:
                    # Find another parent
                    parent2_idx = mating_pool[rng.integers(len(mating_pool))]
                    x1 = X[parent_idx]
                    x2 = X[parent2_idx]

                    eta = 20
                    c1 = np.zeros(dim)
                    c2 = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < 0.5:
                            if abs(x1[j] - x2[j]) < 1e-14:
                                c1[j] = x1[j]
                                c2[j] = x2[j]
                            else:
                                u = rng.random()
                                if u <= 1 / (eta + 1):
                                    beta = (2 * u) ** (1 / (eta + 1))
                                elif u >= 1 - 1 / (eta + 1):
                                    beta = (2 * (1 - u)) ** (1 / (eta + 1))
                                else:
                                    beta = 1.0
                                c1[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                                c2[j] = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
                        else:
                            c1[j] = x1[j]
                            c2[j] = x2[j]

                    # Polynomial mutation
                    for child in [c1, c2]:
                        for j in range(dim):
                            if rng.random() < 1.0 / dim:
                                u = rng.random()
                                delta_q = (1 + 2 * (child[j] - lo[j]) / (hi[j] - lo[j])) ** (-1 / (eta + 1)) if u < 0.5 else \
                                          (1 + 2 * (hi[j] - child[j]) / (hi[j] - lo[j])) ** (-1 / (eta + 1))
                                delta_q = 0.5 * (delta_q ** (eta + 1) - 1)
                                child[j] += delta_q * (hi[j] - lo[j])

                    offspring_X.append(np.clip(c1, lo, hi))
                    if len(offspring_X) < N:
                        offspring_X.append(np.clip(c2, lo, hi))
                else:
                    # Just mutate
                    child = X[parent_idx].copy()
                    for j in range(dim):
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            delta_q = (1 + 2 * (child[j] - lo[j]) / (hi[j] - lo[j])) ** (-1 / (eta + 1)) if u < 0.5 else \
                                      (1 + 2 * (hi[j] - child[j]) / (hi[j] - lo[j])) ** (-1 / (eta + 1))
                            delta_q = 0.5 * (delta_q ** (eta + 1) - 1)
                            child[j] += delta_q * (hi[j] - lo[j])
                    child = np.clip(child, lo, hi)
                    offspring_X.append(child)

            if not offspring_X:
                break

            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Select next generation
            combined_fronts = fast_non_dominated_sort(combined_F)

            next_indices = []
            for front in combined_fronts:
                if len(next_indices) + len(front) <= N:
                    next_indices.extend(front)
                else:
                    # Need to select from this front
                    remaining = N - len(next_indices)
                    dist = crowding_distance(combined_F, front)
                    order = np.argsort(-dist)
                    next_indices.extend([front[i] for i in order[:remaining]])
                    break

            X = combined_X[next_indices]
            F = combined_F[next_indices]

        return X
