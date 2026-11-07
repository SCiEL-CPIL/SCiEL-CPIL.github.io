"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_vars = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size for NSGA-II
        # With budget 200, we can afford 4 generations of size 50 or 2 generations of size 100.
        # Let's use 50 for more generations.
        pop_size = 50

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n_vars))
        F = np.array([problem(x) for x in X])
        remaining_budget = budget - pop_size

        def fast_non_dominated_sort(F):
            """
            Fast non-dominated sorting.
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominates = [0] * n  # number of solutions dominated by i
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[i] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1

            for i in range(n):
                if n_dominates[i] == 0:
                    front0.append(i)

            if front0:
                fronts.append(front0)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1

            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for a front.
            Returns an array of crowding distances for the solutions in the front.
            """
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n

            cd = [0.0] * n
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_indices = sorted(front_indices, key=lambda i: F[i, m])
                # First and last get infinite distance
                cd[front_indices.index(sorted_indices[0])] = float('inf')
                cd[front_indices.index(sorted_indices[-1])] = float('inf')

                # Middle points
                for k in range(1, n - 1):
                    f_range = F[sorted_indices[-1], m] - F[sorted_indices[0], m]
                    if f_range > 0:
                        cd[front_indices.index(sorted_indices[k])] += (
                            F[sorted_indices[k + 1], m] - F[sorted_indices[k - 1], m]
                        ) / f_range

            return cd

        def sbx_crossover(x1, x2, lo, hi, eta=20.0):
            """Simulated Binary Crossover."""
            n = len(x1)
            child = np.zeros_like(x1)
            for i in range(n):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        child[i] = x1[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        child[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child[i] = np.clip(child[i], lo[i], hi[i])
                else:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        child[i] = x1[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        child[i] = 0.5 * ((1 + beta) * x2[i] + (1 - beta) * x1[i])
                        child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        def polynomial_mutation(x, lo, hi, eta=20.0, mutation_prob=0.1):
            """Polynomial Mutation."""
            n = len(x)
            x_mut = x.copy()
            for i in range(n):
                if rng.random() < mutation_prob:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x_mut[i] += delta_q * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main loop
        while remaining_budget >= pop_size:
            # Create offspring population
            offspring_X = np.zeros_like(X)
            offspring_F = np.zeros_like(F)

            # Tournament selection
            def tournament_select(F, fronts, k=2):
                # Assign ranks
                rank = [0] * len(F)
                for r, front in enumerate(fronts):
                    for idx in front:
                        rank[idx] = r

                # Select k random candidates
                candidates = rng.integers(0, len(F), size=k)
                best = candidates[0]
                for c in candidates[1:]:
                    if rank[c] < rank[best]:
                        best = c
                    elif rank[c] == rank[best]:
                        # Tie-break by crowding distance (not computed here, so just pick randomly)
                        if rng.random() < 0.5:
                            best = c
                return best

            for i in range(pop_size):
                if remaining_budget < 2:
                    break

                # Select two parents
                p1 = tournament_select(F, fronts)
                p2 = tournament_select(F, fronts)

                # Crossover
                if rng.random() < 0.9:
                    x_child = sbx_crossover(X[p1], X[p2], lo, hi)
                else:
                    x_child = X[p1].copy()

                # Mutation
                x_child = polynomial_mutation(x_child, lo, hi)

                # Evaluate
                f_child = problem(x_child)
                remaining_budget -= 1

                offspring_X[i] = x_child
                offspring_F[i] = f_child

            # Combine parent and offspring populations
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sorting on combined population
            fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate this front using crowding distance
                    remaining_slots = pop_size - len(next_X)
                    if remaining_slots > 0:
                        front_cd = crowding_distance(combined_F, front)
                        # Sort by crowding distance in descending order
                        sorted_front = sorted(range(len(front)), key=lambda i: front_cd[i], reverse=True)
                        for i in sorted_front[:remaining_slots]:
                            idx = front[i]
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
