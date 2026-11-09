"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary algorithm for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size: balance between exploration and exploitation
        # With budget=200, we can afford a population of ~20-30
        pop_size = min(30, max(10, budget // 8))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_dom = np.zeros(n, dtype=np.int32)  # number of solutions dominated by i
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            # First front
            front = [i for i in range(n) if n_dom[i] == 0]
            if front:
                fronts.append(front)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n, dtype=np.float64)
            F_sub = F[indices]

            for m in range(n_obj):
                # Sort by objective m
                order = np.argsort(F_sub[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf

                f_min = F_sub[order[0], m]
                f_max = F_sub[order[-1], m]
                f_range = f_max - f_min
                if f_range > 1e-15:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / f_range

            return cd

        def tournament_select(F, fronts, pop_size=2):
            """Tournament selection based on rank and crowding distance."""
            # Assign ranks
            n = len(F)
            ranks = np.zeros(n, dtype=np.int32)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distance per front
            cd = np.zeros(n, dtype=np.float64)
            for front in fronts:
                cd[front] = crowding_distance(F, front)

            # Tournament
            best = None
            for _ in range(pop_size):
                idx = rng.integers(n)
                if best is None:
                    best = idx
                else:
                    # Compare: lower rank is better, then higher crowding distance
                    if ranks[idx] < ranks[best]:
                        best = idx
                    elif ranks[idx] == ranks[best] and cd[idx] > cd[best]:
                        best = idx
            return best

        def sbx_crossover(x1, x2, lo, hi):
            """Simulated binary crossover."""
            eta = 20.0
            if rng.random() < 0.5:
                x1, x2 = x2, x1

            child1 = np.empty_like(x1)
            child2 = np.empty_like(x2)

            for i in range(dim):
                if abs(x1[i] - x2[i]) < 1e-15:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                    continue

                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

                c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])

                child1[i] = np.clip(c1, lo[i], hi[i])
                child2[i] = np.clip(c2, lo[i], hi[i])

            return child1, child2

        def polynomial_mutation(x, lo, hi):
            """Polynomial mutation."""
            eta = 20.0
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1))

                    x_mut[i] = x[i] + delta * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Generate offspring
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                # Select two parents via tournament
                p1_idx = tournament_select(F, fronts)
                p2_idx = tournament_select(F, fronts)

                # Crossover
                if rng.random() < 0.9:  # crossover probability
                    child1, child2 = sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                else:
                    child1 = X[p1_idx].copy()
                    child2 = X[p2_idx].copy()

                # Mutation
                child1 = polynomial_mutation(child1, lo, hi)
                child2 = polynomial_mutation(child2, lo, hi)

                # Evaluate
                f1 = problem(child1)
                used += 1
                offspring_X.append(child1)
                offspring_F.append(f1)

                if used >= budget:
                    break

                if time.time() > deadline:
                    break

                f2 = problem(child2)
                used += 1
                offspring_X.append(child2)
                offspring_F.append(f2)

                if used >= budget:
                    break

                if time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F, dtype=np.float64)
            offspring_X = np.array(offspring_X, dtype=np.float64)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sorting on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Assign ranks
            n_combined = len(combined_F)
            ranks = np.zeros(n_combined, dtype=np.int32)
            for rank, front in enumerate(combined_fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distance per front
            cd = np.zeros(n_combined, dtype=np.float64)
            for front in combined_fronts:
                cd[front] = crowding_distance(combined_F, front)

            # Select next population
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    front_indices = np.array(front)
                    front_cd = cd[front_indices]
                    order = np.argsort(front_cd)[::-1]  # descending
                    for i in range(remaining):
                        idx = front_indices[order[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            if not next_X:
                break

            X = np.array(next_X, dtype=np.float64)
            F = np.array(next_F, dtype=np.float64)

        return X
# EVOLVE-BLOCK-END
