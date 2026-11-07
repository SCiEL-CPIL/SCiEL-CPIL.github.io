"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Population size - increased for better initial coverage
        pop_size = 30

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (all objectives <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Perform non-dominated sorting, return list of fronts."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        dominated_set[i].append(j)
                    elif dominates(F[j], F[i]):
                        domination_count[i] += 1

            # First front
            front = [i for i in range(n) if domination_count[i] == 0]
            fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)

            m = self.n_obj
            cd = np.zeros(len(indices))

            for k in range(m):
                obj_vals = F[indices, k]
                sorted_idx = np.argsort(obj_vals)
                obj_sorted = obj_vals[sorted_idx]

                min_val = obj_sorted[0]
                max_val = obj_sorted[-1]
                if max_val - min_val < 1e-12:
                    continue

                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                for i in range(1, len(sorted_idx) - 1):
                    cd[sorted_idx[i]] += (obj_sorted[i + 1] - obj_sorted[i - 1]) / (max_val - min_val)

            return cd

        # Main optimization loop - generate 10 children per iteration
        n_children = 10
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Adaptive mutation rate - wider exploration early, refinement later
            progress = used / self.budget
            mutation_scale = 0.3 * (1 - progress) + 0.05 * progress

            # Get non-dominated fronts
            fronts = non_dominated_sort(F)

            # Select parents from first front(s)
            parent_pool = []
            for front in fronts:
                if len(parent_pool) + len(front) >= pop_size:
                    parent_pool.extend(front)
                    break
                parent_pool.extend(front)

            # If we still need more, add from next front
            if len(parent_pool) < pop_size:
                for front in fronts:
                    for idx in front:
                        if idx not in parent_pool:
                            parent_pool.append(idx)
                        if len(parent_pool) >= pop_size:
                            break
                    if len(parent_pool) >= pop_size:
                        break

            # Generate multiple children per iteration
            new_children_X = []
            new_children_F = []

            for _ in range(n_children):
                if used >= self.budget or time.time() > deadline:
                    break

                # Select 2 parents via tournament
                def tournament_select(pool, F):
                    candidates = [pool[rng.integers(len(pool))] for _ in range(2)]
                    best = min(candidates, key=lambda i: F[i].sum())
                    return best

                p1_idx = tournament_select(parent_pool, F)
                p2_idx = tournament_select(parent_pool, F)

                # Crossover
                if rng.random() < 0.9:
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            child[d] = X[p1_idx, d]
                        else:
                            child[d] = X[p2_idx, d]
                else:
                    child = X[p1_idx].copy()

                # Add mutation
                child += mutation_scale * (hi - lo) * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                # Evaluate child
                child_F = np.array(problem(child))
                used += 1

                new_children_X.append(child)
                new_children_F.append(child_F)

            if not new_children_X:
                break

            # Combine and select next generation
            combined_X = np.vstack([X, np.array(new_children_X)])
            combined_F = np.vstack([F, np.array(new_children_F)])

            # Non-dominated sort on combined population
            combined_fronts = non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate this front
                    needed = pop_size - len(next_X)
                    if needed > 0:
                        # Sort by crowding distance
                        cd = crowding_distance(combined_F, front)
                        sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                        for idx in sorted_front[:needed]:
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
