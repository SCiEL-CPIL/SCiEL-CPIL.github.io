# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(100, max(20, self.budget // 2))
        # Ensure we can do at least one generation
        if pop_size * 2 > self.budget:
            pop_size = max(10, self.budget // 2)

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: fast non-dominated sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
                if n_dom[p] == 0:
                    fronts[0].append(p)
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            return fronts[:-1]  # remove empty last

        # Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            m = self.n_obj
            cd = np.zeros(len(indices))
            for k in range(m):
                f = F[indices, k]
                sorted_idx = np.argsort(f)
                cd_sorted = np.zeros(len(indices))
                cd_sorted[0] = np.inf
                cd_sorted[-1] = np.inf
                f_min, f_max = f[sorted_idx[0]], f[sorted_idx[-1]]
                if f_max == f_min:
                    continue
                for i in range(1, len(indices) - 1):
                    cd_sorted[i] += (f[sorted_idx[i + 1]] - f[sorted_idx[i - 1]]) / (f_max - f_min)
                # map back
                for i, idx in enumerate(sorted_idx):
                    cd[indices.index(idx)] = cd_sorted[i]
            return cd

        # Binary tournament selection
        def tournament_select(X, F, fronts):
            # Assign rank
            rank = np.zeros(len(X), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            # Crowding distance per front
            cd = np.zeros(len(X))
            for front in fronts:
                if len(front) > 0:
                    cd[front] = crowding_distance(F, front)

            def select_one():
                i = rng.integers(len(X))
                j = rng.integers(len(X))
                if rank[i] < rank[j]:
                    return i
                elif rank[j] < rank[i]:
                    return j
                else:
                    return i if cd[i] > cd[j] else j

            return select_one(), select_one()

        # SBX crossover
        def sbx_crossover(x1, x2, eta=15.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if rng.random() <= 1.0 / (eta + 1):
                            beta = (2 * rng.random()) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2.0 * rng.random())) ** (1.0 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Polynomial mutation
        def polynomial_mutate(x, eta=20.0):
            y = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    delta1 = (rng.random() * 2) ** (1.0 / (eta + 1)) - 1
                    delta2 = 1 - (rng.random() * 2) ** (1.0 / (eta + 1))
                    y[i] += (hi[i] - lo[i]) * (delta1 + delta2)
            return np.clip(y, lo, hi)

        # Main loop
        while used < self.budget:
            # Create offspring
            offspring = []
            offspring_F = []
            num_offspring = min(pop_size, self.budget - used)

            for _ in range(num_offspring):
                p1, p2 = tournament_select(X, F, fast_non_dominated_sort(F))
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                offspring.append(c1)
                offspring.append(c2)
                if used + len(offspring) > self.budget:
                    break

            # Evaluate offspring
            for c in offspring:
                if used >= self.budget:
                    break
                fc = problem(c)
                offspring_F.append(fc)
                used += 1

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_F = np.array(offspring_F)

            # Combine parents and offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            fronts = fast_non_dominated_sort(F_all)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_all, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
