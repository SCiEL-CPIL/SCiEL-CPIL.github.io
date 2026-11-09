"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = 40
        # Number of offspring per generation
        n_offspring = 20

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts (list of indices) using fast non-dominated sorting."""
            m = F.shape[0]
            n = F.shape[1]
            S = [[] for _ in range(m)]
            n_dom = np.zeros(m, dtype=int)
            fronts = []
            current_front = []

            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1

            for i in range(m):
                if n_dom[i] == 0:
                    current_front.append(i)

            fronts.append(current_front)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            # Remove the last empty front
            if fronts and len(fronts[-1]) == 0:
                fronts.pop()

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distances for indices in front_indices."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            F_sub = F[front_indices]

            for m in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / f_range

            return dist

        def select_parents(F, X, n_select):
            """Select n_select parents using tournament selection based on non-dominated rank and crowding distance."""
            m = F.shape[0]
            fronts = fast_non_dominated_sort(F)

            # Assign ranks
            rank = np.zeros(m, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Compute crowding distance for the first front
            # For simplicity, just use rank for selection
            # Select using tournament on rank
            selected = []
            for _ in range(n_select):
                # Tournament of 2
                i, j = rng.integers(0, m, size=2)
                if rank[i] < rank[j]:
                    selected.append(i)
                elif rank[j] < rank[i]:
                    selected.append(j)
                else:
                    # Same rank, pick randomly
                    selected.append(rng.integers(0, m))
            return selected

        def crossover(parent1, parent2):
            """Simulated binary crossover."""
            child1 = parent1.copy()
            child2 = parent2.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    # SBX
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / 20)
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / 20)
                    c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                    child1[i] = c1
                    child2[i] = c2
            return child1, child2

        def mutation(x):
            """Polynomial mutation."""
            child = x.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / 20) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / 20)
                    child[i] += delta_q * (hi[i] - lo[i])
            return np.clip(child, lo, hi)

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            n_off = min(n_offspring, budget - used)
            children_X = []
            children_F = []

            for _ in range(n_off):
                # Select parents
                p1_idx, p2_idx = rng.integers(0, pop_size, size=2)
                # Ensure they are different
                if p1_idx == p2_idx:
                    p2_idx = (p1_idx + 1) % pop_size

                c1, c2 = crossover(X[p1_idx], X[p2_idx])
                c1 = mutation(c1)
                c2 = mutation(c2)

                if used + 1 <= budget:
                    fc1 = problem(c1)
                    used += 1
                    children_X.append(c1)
                    children_F.append(fc1)

                if used + 1 <= budget:
                    fc2 = problem(c2)
                    used += 1
                    children_X.append(c2)
                    children_F.append(fc2)

                # Check deadline
                if time.time() > deadline:
                    break

            if not children_X:
                break

            # Combine parent and offspring
            X_combined = np.vstack([X, np.array(children_X)])
            F_combined = np.vstack([F, np.array(children_F)])

            # Selection: keep best pop_size individuals
            m = F_combined.shape[0]
            fronts = fast_non_dominated_sort(F_combined)

            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to use crowding distance to select remaining
                    remaining = pop_size - len(new_X)
                    if remaining > 0:
                        cd = crowding_distance(F_combined, front)
                        sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                        for i in range(remaining):
                            idx = front[sorted_front[i]]
                            new_X.append(X_combined[idx])
                            new_F.append(F_combined[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

            # Check deadline
            if time.time() > deadline:
                break

        return X
# EVOLVE-BLOCK-END
