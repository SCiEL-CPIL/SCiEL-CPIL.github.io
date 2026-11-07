"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2."""

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

        # Population size
        pop_size = 100
        n_obj = self.n_obj
        dim = self.dim

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
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

        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = np.zeros_like(X)
            offspring_F = np.zeros_like(F)

            for i in range(pop_size):
                if time.time() > deadline:
                    break

                # Select parents using tournament selection
                parents = []
                for _ in range(2):
                    # Find non-dominated fronts
                    fronts = non_dominated_sort(F)
                    # Pick from best front
                    best_front = fronts[0]
                    if len(best_front) > 1:
                        cd = crowding_distance(F, best_front)
                        # Tournament on crowding distance
                        candidates = list(range(len(best_front)))
                        winner = max(candidates, key=lambda x: cd[x])
                        parent_idx = best_front[winner]
                    else:
                        parent_idx = best_front[0]
                    parents.append(parent_idx)

                # SBX crossover
                if rng.random() < 0.9:
                    child1 = X[parents[0]].copy()
                    child2 = X[parents[1]].copy()
                    for d in range(dim):
                        if rng.random() < 0.5:
                            if rng.random() < 0.5:
                                child1[d], child2[d] = child2[d], child1[d]
                else:
                    child1 = X[parents[0]].copy()
                    child2 = X[parents[1]].copy()

                # Polynomial mutation
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        delta = 0.1 * (hi[d] - lo[d])
                        child1[d] += delta * rng.standard_normal()
                    if rng.random() < 1.0 / dim:
                        delta = 0.1 * (hi[d] - lo[d])
                        child2[d] += delta * rng.standard_normal()

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                offspring_X[i] = child1
                offspring_F[i] = problem(child1)
                used += 1

                if used < self.budget and time.time() < deadline:
                    offspring_X[i + pop_size // 2 if i < pop_size // 2 else i] = child2
                    # Actually, let's just do one child per iteration to keep it simple
                    # and adjust population size accordingly
                    break

            # Actually, let's simplify: generate one child per iteration
            # and replace the worst individual
            # This is more like the original but with better selection

            # Let me redo this with a cleaner approach
            # Generate offspring population of size pop_size
            # But we need to be careful about budget

            # For simplicity, let's generate one child per loop iteration
            # and use non-dominated sorting for parent selection
            # This keeps the budget usage predictable

            # Select two parents from non-dominated fronts
            fronts = non_dominated_sort(F)
            parent1_idx = parent2_idx = None

            # Pick from first front
            if len(fronts[0]) > 0:
                if len(fronts[0]) > 1:
                    cd1 = crowding_distance(F, fronts[0])
                    parent1_idx = fronts[0][np.argmax(cd1)]
                    cd2 = crowding_distance(F, fronts[0])
                    parent2_idx = fronts[0][np.argmax(cd2)]
                else:
                    parent1_idx = parent2_idx = fronts[0][0]

            if parent1_idx is None:
                parent1_idx = parent2_idx = 0

            # Create child via mutation
            child = X[parent1_idx].copy()
            for d in range(dim):
                if rng.random() < 1.0 / dim:
                    delta = 0.05 * (hi[d] - lo[d])
                    child[d] += delta * rng.standard_normal()
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst individual based on dominance and crowding
            # Find the individual with worst crowding distance in worst front
            fronts_all = non_dominated_sort(F)
            worst_front = fronts_all[-1]
            if len(worst_front) > 1:
                cd = crowding_distance(F, worst_front)
                worst_idx = worst_front[np.argmin(cd)]
            else:
                worst_idx = worst_front[0] if worst_front else 0

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
