"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: NSGA-II style with adaptive mutation and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Initial population
        pop_size = min(40, self.budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_mask(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                # Check if any other point dominates i
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated

        def crowding_distance(F, mask):
            idx = np.where(mask)[0]
            n = len(idx)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[idx, m]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                min_v, max_v = vals[order[0]], vals[order[-1]]
                if max_v - min_v > 0:
                    for k in range(1, n - 1):
                        cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_v - min_v)
            return cd

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated front
            nd_mask = non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                # All dominated, pick random
                parent_idx = rng.integers(len(F))
            else:
                # Select parent from non-dominated set using crowding distance
                cd = crowding_distance(F, nd_mask)
                parent_idx = nd_indices[rng.choice(len(nd_indices), p=cd / cd.sum())]

            # Adaptive mutation: start with larger step, decay over time
            progress = used / self.budget
            step_size = 0.3 * (1.0 - progress) + 0.05
            child = X[parent_idx] + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point in population
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
