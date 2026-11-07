"""Multi-objective optimizer on WFG9 using NSGA-II."""
# EVOLVE-BLOCK-START
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
        deadline = start_time + 290  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Population size: use a good chunk of budget
        pop_size = min(40, max(20, self.budget // 5))

        # Initialize population
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best archive (non-dominated points)
        # We'll maintain the full population and use NSGA-II selection

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (indices)."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)  # number of solutions dominated by i
            dominated_set = [[] for _ in range(n)]    # solutions dominated by i
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1

            # First front: solutions with domination_count == 0
            current_front = [i for i in range(n) if domination_count[i] == 0]
            if current_front:
                fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[front_indices, m]
                min_val = f_vals.min()
                max_val = f_vals.max()
                if max_val - min_val < 1e-14:
                    dist[:] = np.inf
                    continue
                # Sort by objective m
                order = np.argsort(f_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (f_vals[order[k + 1]] - f_vals[order[k - 1]]) / (max_val - min_val)
            return dist

        def tournament_select(parents, F, fronts, k=2):
            """Binary tournament selection based on rank and crowding distance."""
            n = len(parents)
            i = np.random.randint(n)
            j = np.random.randint(n)
            # Compare ranks
            rank_i = next(idx for idx, fr in enumerate(fronts) if i in fr)
            rank_j = next(idx for idx, fr in enumerate(fronts) if j in fr)
            if rank_i < rank_j:
                return i
            elif rank_j < rank_i:
                return j
            else:
                # Same rank, use crowding distance
                fr_idx = rank_i
                front = fronts[fr_idx]
                cd = crowding_distance(F, front)
                # Map parent indices to positions in front
                pos_i = front.index(i)
                pos_j = front.index(j)
                if cd[pos_i] > cd[pos_j]:
                    return i
                else:
                    return j

        def crossover(x1, x2):
            """BLX-alpha style crossover."""
            if np.random.random() < 0.5:
                return (x1 + x2) / 2.0
            alpha = 0.5
            lo_c = np.minimum(x1, x2) - alpha * np.abs(x1 - x2)
            hi_c = np.maximum(x1, x2) + alpha * np.abs(x1 - x2)
            return lo_c + (hi_c - lo_c) * np.random.random(dim)

        def mutate(x, sigma=0.1):
            """Gaussian mutation with adaptive sigma."""
            return x + sigma * span * np.random.standard_normal(dim)

        # Main evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Compute fronts and crowding distances
            fronts = fast_non_dominated_sort(F)

            # Create offspring
            n_offspring = pop_size
            offspring_X = []
            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                # Select parents via tournament
                p1 = tournament_select(np.arange(pop_size), F, fronts)
                p2 = tournament_select(np.arange(pop_size), F, fronts)
                # Crossover
                child = crossover(X[p1], X[p2])
                # Mutate
                child = mutate(child, sigma=0.1)
                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += len(offspring_X)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_X)

            # Sort combined by non-dominated fronts
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(combined_X[front])
                    new_F.append(combined_F[front])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    # Select top 'remaining' by crowding distance
                    order = np.argsort(-cd)
                    selected = [front[i] for i in order[:remaining]]
                    new_X.append(combined_X[selected])
                    new_F.append(combined_F[selected])
                    break

            if new_X:
                X = np.vstack(new_X)
                F = np.vstack(new_F)
            else:
                break

        return X
# EVOLVE-BLOCK-END
