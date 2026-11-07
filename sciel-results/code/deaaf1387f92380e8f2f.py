"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with adaptive mutation and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 2))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            front = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if dominates(F[i], F[j]):
                            S[i].append(j)
                        elif dominates(F[j], F[i]):
                            n_count[i] += 1
                if n_count[i] == 0:
                    front.append(i)

            fronts = []
            while front:
                fronts.append(front[:])
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n == 0:
                return np.array([])
            dist = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort(F[indices, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_min = F[indices[sorted_idx[0]], m]
                f_max = F[indices[sorted_idx[-1]], m]
                if f_max > f_min:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (F[indices[sorted_idx[k + 1]], m] - F[indices[sorted_idx[k - 1]], m]) / (f_max - f_min)
            return dist

        # Track best hypervolume contribution for adaptation
        def hypervolume_contribution(F, ref):
            """Approximate HV contribution by summing dominated volumes (simplified)."""
            # Use a simple approximation: sum of (ref - f) for points in first front
            return np.sum(np.prod(ref - F, axis=1))

        ref = np.array([2.5, 2.5, 2.5])
        best_hv = hypervolume_contribution(F, ref)
        no_improve_count = 0
        mutation_scale = 0.1  # Initial mutation scale

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort for selection
            fronts = fast_non_dominated_sort(F)

            # Select two parents from the best front using crowding distance
            if len(fronts[0]) > 1:
                cd = crowding_distance(F, fronts[0])
                parent_indices = list(range(len(fronts[0])))
                parent_indices.sort(key=lambda x: cd[x], reverse=True)
                p1 = fronts[0][parent_indices[0]]
                top_half = parent_indices[:max(1, len(parent_indices) // 2)]
                p2 = fronts[0][top_half[rng.integers(len(top_half))]]
            elif len(fronts[0]) == 1:
                p1 = p2 = fronts[0][0]
            else:
                p1 = p2 = 0

            # Crossover
            if rng.random() < 0.9:
                alpha = rng.random()
                child = (1 - alpha) * X[p1] + alpha * X[p2]
            else:
                child = X[p1].copy()

            # Adaptive mutation: scale based on stagnation
            current_hv = hypervolume_contribution(F, ref)
            if current_hv > best_hv:
                best_hv = current_hv
                no_improve_count = 0
                mutation_scale = max(0.05, mutation_scale * 0.9)
            else:
                no_improve_count += 1
                if no_improve_count % 10 == 0:
                    mutation_scale = min(0.3, mutation_scale * 1.2)

            for d in range(dim):
                if rng.random() < 1.0 / dim:
                    delta = mutation_scale * (hi[d] - lo[d])
                    child[d] += delta * rng.standard_normal()

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst individual
            fronts_all = fast_non_dominated_sort(F)
            worst_front = fronts_all[-1]
            if len(worst_front) > 1:
                cd_worst = crowding_distance(F, worst_front)
                worst_idx = worst_front[np.argmin(cd_worst)]
            else:
                worst_idx = worst_front[0] if worst_front else 0

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
