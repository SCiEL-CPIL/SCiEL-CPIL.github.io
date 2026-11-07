"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with deadline guard, crossover, and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_mask(self, F):
        """Return boolean mask of non-dominated points in F."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            # Check if any other point dominates i
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        pop_size = int(min(20, budget // 4))
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive mutation scale
        mut_scale = 0.1

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_mask = self._non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) >= 2:
                # Crossover: pick two parents from non-dominated set
                i1, i2 = rng.choice(nd_indices, size=2, replace=False)
                alpha = rng.random(dim)
                child = alpha * X[i1] + (1 - alpha) * X[i2]
            else:
                # Mutation from single parent
                if len(nd_indices) == 1:
                    parent_idx = nd_indices[0]
                else:
                    parent_idx = rng.integers(len(X))
                child = X[parent_idx].copy()

            # Adaptive mutation
            child = child + mut_scale * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replacement: remove the point most dominated (highest dominance count)
            # Compute dominance count for each point
            dom_count = np.zeros(len(F), dtype=int)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dom_count[i] += 1

            worst = int(np.argmax(dom_count))
            if dom_count[worst] == 0:
                # All non-dominated; remove the one with worst sum
                worst = int(np.argmax(np.sum(F, axis=1)))

            X[worst] = child
            F[worst] = fc

            # Adapt mutation scale: decrease if improving, increase if stuck
            if used % 20 == 0:
                if np.min(np.sum(F, axis=1)) < np.min(np.sum(F[:used - 20] if used - 20 > 0 else F, axis=1)):
                    mut_scale = max(0.01, mut_scale * 0.9)
                else:
                    mut_scale = min(0.3, mut_scale * 1.1)

        return X
# EVOLVE-BLOCK-END
