"""Multi-objective optimizer for WFG7 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        scale = hi - lo

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = 40
        # Crossover and mutation parameters
        cx_rate = 0.9
        mut_rate = 1.0 / dim
        sigma = 0.2  # mutation step size relative to scale

        def dominates(a, b):
            """Return True if a dominates b (a is better or equal in all, better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Perform fast non-dominated sorting. Returns list of fronts (each is a list of indices)."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)  # number of solutions that dominate i
            dominated_set = [[] for _ in range(n)]  # set of solutions dominated by i
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif dominates(F[j], F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            # First front
            front = []
            for i in range(n):
                if domination_count[i] == 0:
                    front.append(i)
            if front:
                fronts.append(front)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                    k += 1
                else:
                    break

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices with objectives F."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            F_sub = F[indices]

            for m in range(n_obj):
                # Sort by objective m
                order = np.argsort(F_sub[:, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf

                f_range = F_sub[order[-1], m] - F_sub[order[0], m]
                if f_range == 0:
                    continue

                for i in range(1, n - 1):
                    dist[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / f_range

            return dist

        def crossover(x1, x2):
            """SBX-like crossover."""
            if np.random.random() > cx_rate:
                return x1.copy(), x2.copy()

            child1 = x1.copy()
            child2 = x2.copy()
            for d in range(dim):
                if np.random.random() < 0.5:
                    child1[d], child2[d] = x2[d], x1[d]
            return child1, child2

        def mutate(x):
            """Gaussian mutation."""
            if np.random.random() < mut_rate:
                x = x + sigma * scale * np.random.randn(dim)
            return np.clip(x, lo, hi)

        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + scale * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for final hypervolume
        all_X = X.copy()
        all_F = F.copy()

        # Evolution loop
        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring population
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size and used < budget:
                if time.time() > deadline:
                    break

                # Tournament selection
                i1 = self._tournament_select(F, pop_size)
                i2 = self._tournament_select(F, pop_size)

                # Crossover
                c1, c2 = crossover(X[i1], X[i2])

                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)

                # Evaluate
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)

                if used >= budget:
                    break

                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sorting
            fronts = non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            front_idx = 0

            while len(next_X) < pop_size and front_idx < len(fronts):
                front = fronts[front_idx]
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    front_idx += 1
                else:
                    # Need to fill partially from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)  # descending crowding distance
                    for k in range(remaining):
                        idx = front[order[k]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

            # Record all points for hypervolume
            all_X = np.vstack([all_X, offspring_X])
            all_F = np.vstack([all_F, offspring_F])

        return all_X

    def _tournament_select(self, F, k=2):
        """Binary tournament selection based on non-dominated rank."""
        n = len(F)
        i = np.random.randint(n)
        j = np.random.randint(n)

        # Compute ranks quickly: use domination check
        # For efficiency, we use a simple heuristic:
        # If i dominates j, pick i; if j dominates i, pick j; else pick randomly
        if dominates(F[i], F[j]):
            return i
        elif dominates(F[j], F[i]):
            return j
        else:
            # Non-dominated pair: use a simple metric (sum of objectives)
            # Lower sum is better (proxy for closeness to ideal)
            if F[i].sum() <= F[j].sum():
                return i
            else:
                return j
# EVOLVE-BLOCK-END
