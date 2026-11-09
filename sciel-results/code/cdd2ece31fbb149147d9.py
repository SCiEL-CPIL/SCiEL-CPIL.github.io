"""Multi-objective optimizer for RE21 using a population-based evolutionary strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a simple NSGA-II-like approach with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Population size: use a reasonable fraction of budget
        pop_size = min(50, max(10, self.budget // 4))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Store all evaluated points for hypervolume computation
        all_X = [X.copy()]
        all_F = [F.copy()]

        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Find non-dominated solutions
            non_dom_idx = self._find_non_dominated(F)

            # Select parents from non-dominated solutions
            if len(non_dom_idx) > 0:
                parents_idx = non_dom_idx
            else:
                parents_idx = np.arange(len(F))

            # Generate offspring
            n_offspring = pop_size
            offspring_X = np.empty((n_offspring, self.dim))
            offspring_F = np.empty((n_offspring, self.n_obj))

            for i in range(n_offspring):
                if time.time() > deadline - 5:
                    break

                # Select two parents (tournament-like: pick randomly from non-dominated)
                p1_idx = parents_idx[rng.integers(len(parents_idx))]
                p2_idx = parents_idx[rng.integers(len(parents_idx))]

                # SBX crossover
                if rng.random() < 0.9:
                    child = self._sbx_crossover(X[p1_idx], X[p2_idx], lo, hi, rng)
                else:
                    child = X[p1_idx].copy()

                # Mutation
                child = self._polynomial_mutation(child, lo, hi, rng)

                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

            if used >= self.budget:
                break

            # Combine parent and offspring populations
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Select next generation using non-dominated sorting
            X, F = self._select_next_generation(combined_X, combined_F, pop_size)

            # Record evaluated points
            all_X.append(X.copy())
            all_F.append(F.copy())

        return X

    def _find_non_dominated(self, F):
        """Find indices of non-dominated solutions."""
        n = len(F)
        non_dom = np.ones(n, dtype=bool)
        for i in range(n):
            if not non_dom[i]:
                continue
            for j in range(n):
                if i == j or not non_dom[j]:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    non_dom[i] = False
                    break
        return np.where(non_dom)[0]

    def _sbx_crossover(self, x1, x2, lo, hi, rng):
        """Simulated Binary Crossover."""
        eta = 20.0
        child = np.empty_like(x1)
        for k in range(len(x1)):
            if rng.random() < 0.5:
                if abs(x1[k] - x2[k]) > 1e-14:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[k] = 0.5 * ((1 + beta) * x1[k] + (1 - beta) * x2[k])
                else:
                    child[k] = x1[k]
            else:
                if abs(x1[k] - x2[k]) > 1e-14:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[k] = 0.5 * ((1 + beta) * x2[k] + (1 - beta) * x1[k])
                else:
                    child[k] = x2[k]
        return np.clip(child, lo, hi)

    def _polynomial_mutation(self, x, lo, hi, rng):
        """Polynomial Mutation."""
        eta = 20.0
        mutated = x.copy()
        for k in range(len(x)):
            if rng.random() < 0.5:
                u = rng.random()
                if u < 0.5:
                    xy = (2 * u + (x[k] - lo[k]) / (hi[k] - lo[k])) ** (1.0 / (eta + 1))
                    delta = xy - 1
                else:
                    xy = (2 * (1 - u) + (hi[k] - x[k]) / (hi[k] - lo[k])) ** (1.0 / (eta + 1))
                    delta = 1 - xy
                mutated[k] = x[k] + delta * (hi[k] - lo[k])
        return np.clip(mutated, lo, hi)

    def _select_next_generation(self, X, F, pop_size):
        """Select next generation using non-dominated sorting and crowding distance."""
        n = len(X)
        if n <= pop_size:
            return X, F

        # Non-dominated sorting
        fronts = []
        remaining = np.arange(n)
        while len(remaining) > 0:
            non_dom = self._find_non_dominated(F[remaining])
            front = remaining[non_dom]
            fronts.append(front)
            remaining = np.setdiff1d(remaining, front)

        # Fill next generation
        next_idx = []
        for front in fronts:
            if len(next_idx) + len(front) <= pop_size:
                next_idx.extend(front)
            else:
                # Need to select from this front based on crowding distance
                needed = pop_size - len(next_idx)
                if needed > 0:
                    # Compute crowding distance for this front
                    front_F = F[front]
                    distances = self._crowding_distance(front_F)
                    # Select top 'needed' by distance
                    sorted_idx = np.argsort(distances)[::-1]
                    selected = front[sorted_idx[:needed]]
                    next_idx.extend(selected)
                break

        next_idx = np.array(next_idx)
        return X[next_idx], F[next_idx]

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of objective values."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)

        distances = np.zeros(n)
        for m in range(F.shape[1]):
            sorted_idx = np.argsort(F[:, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            if F[sorted_idx[-1], m] - F[sorted_idx[0], m] > 1e-14:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]
                    ) / (F[sorted_idx[-1], m] - F[sorted_idx[0], m])
        return distances
# EVOLVE-BLOCK-END
