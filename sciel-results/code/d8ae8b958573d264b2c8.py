# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_ = hi - lo

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size: keep it manageable
        pop_size = min(30, max(10, budget // 6))

        # Initial population: spread across bounds
        X = lo + rng_ * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_mask(F):
            """Return boolean mask: True if point is non-dominated."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                # Check if any other point dominates i
                dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                dominates_i[i] = False  # exclude self
                if np.any(dominates_i):
                    mask[i] = False
            return mask

        def crowding_distance(F, nd_mask):
            """Compute crowding distance for non-dominated points."""
            n = len(F)
            cd = np.zeros(n)
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) <= 2:
                cd[nd_indices] = np.inf
                return cd
            for m in range(n_obj):
                sorted_idx = nd_indices[np.argsort(F[nd_indices, m])]
                if sorted_idx.size > 0:
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    f_range = F[sorted_idx[-1], m] - F[sorted_idx[0], m]
                    if f_range > 1e-12:
                        for j in range(1, sorted_idx.size - 1):
                            cd[sorted_idx[j]] += (
                                F[sorted_idx[j + 1], m] - F[sorted_idx[j - 1], m]
                            ) / f_range
            return cd

        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated front
            nd_mask = non_dominated_mask(F)
            n_nd = np.sum(nd_mask)

            # If all are non-dominated, use crowding distance to select parents
            if n_nd == len(F):
                cd = crowding_distance(F, nd_mask)
                # Select 2 parents with highest crowding distance
                order = np.argsort(-cd)
                p1 = order[rng.integers(len(order))]
                p2 = order[rng.integers(len(order))]
            else:
                # Select from non-dominated
                nd_indices = np.where(nd_mask)[0]
                p1 = nd_indices[rng.integers(len(nd_indices))]
                p2 = nd_indices[rng.integers(len(nd_indices))]

            # Crossover: blend
            alpha = rng.random(dim)
            child = alpha * X[p1] + (1 - alpha) * X[p2]

            # Mutation
            mut_rate = 0.3
            mut_mask = rng.random(dim) < mut_rate
            child[mut_mask] += 0.15 * rng_ * rng.standard_normal(np.sum(mut_mask))
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point (by sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
