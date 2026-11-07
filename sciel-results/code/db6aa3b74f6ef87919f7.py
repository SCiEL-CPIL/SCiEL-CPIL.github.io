"""Multi-objective optimizer for RE37: Archive-based Differential Evolution with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Archive-based Multi-Objective Differential Evolution for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        # Population size: balance exploration and exploitation
        pop_size = min(80, max(20, budget // 3))
        if pop_size > budget:
            pop_size = budget

        # Initialize population uniformly
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated solutions (for diversity maintenance)
        # We'll maintain the population itself as a dynamic archive

        # Adaptive mutation factor
        F_mu = 0.8  # DE mutation factor
        CR = 0.7    # Crossover rate

        # Track best for return
        best_x = X[0].copy()
        best_f = F[0].copy()

        while used < budget:
            if time.time() > deadline:
                break

            # Generate offspring using differential evolution
            for _ in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct parents
                i = rng.integers(0, pop_size)
                j, k = rng.choice(pop_size, size=2, replace=False)
                while j == i:
                    j = rng.integers(0, pop_size)
                while k == i or k == j:
                    k = rng.integers(0, pop_size)

                # Differential mutation
                v = X[i] + F_mu * (X[j] - X[k])
                v = np.clip(v, lo, hi)

                # Binomial crossover
                r = rng.random()
                j_rand = rng.integers(0, dim)
                child = np.where(rng.random(dim) < CR or np.arange(dim) == j_rand, v, X[i])
                child = np.clip(child, lo, hi)

                # Evaluate child
                f_child = problem(child)
                used += 1

                # Replacement: keep child if it dominates parent or is non-dominated
                # and has better crowding (approximated by distance to reference)
                ref = np.array([1.1, 1.3, 1.1])
                dominates_parent = np.all(f_child <= F[i]) and np.any(f_child < F[i])
                dominated_by_parent = np.all(F[i] <= f_child) and np.any(F[i] < f_child)

                if dominates_parent:
                    X[i] = child
                    F[i] = f_child
                elif not dominated_by_parent:
                    # Non-dominated: compare by distance to reference point
                    # (proxy for hypervolume contribution)
                    dist_child = np.sum(np.maximum(f_child - ref, 0) ** 2)
                    dist_parent = np.sum(np.maximum(F[i] - ref, 0) ** 2)
                    if dist_child < dist_parent:
                        X[i] = child
                        F[i] = f_child

        # Final cleanup: ensure we return something valid
        if F.shape[0] > 0:
            # Find a representative best point (closest to ideal)
            ideal = np.min(F, axis=0)
            dists = np.sum((F - ideal) ** 2, axis=1)
            best_idx = int(np.argmin(dists))
            best_x = X[best_idx].copy()
            best_f = F[best_idx].copy()

        return X
# EVOLVE-BLOCK-END
