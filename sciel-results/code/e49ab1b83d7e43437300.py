# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using adaptive mutation with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        # Initial population: 30 random points
        init_pop = 30
        X = lo + (hi - lo) * rng.random((init_pop, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = init_pop

        # Adaptive mutation scale: start large, shrink over time
        n_children_per_iter = 10
        total_iters = max(1, (budget - init_pop) // n_children_per_iter)

        for it in range(total_iters):
            # Check time limit
            if time.time() > deadline - 5:
                break

            # Progress from 0 to 1
            progress = it / max(1, total_iters - 1)
            # Mutation scale: 0.3 early -> 0.05 late
            mut_scale = 0.3 - 0.25 * progress

            # Find non-dominated solutions
            non_dom_mask = self._non_dominated_mask(F)
            if np.any(non_dom_mask):
                parents = X[non_dom_mask]
                F_parents = F[non_dom_mask]
            else:
                parents = X
                F_parents = F

            # Generate children
            for _ in range(n_children_per_iter):
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Select a parent (prefer non-dominated)
                p_idx = rng.integers(len(parents))
                parent = parents[p_idx]

                # Mutate with adaptive scale
                child = parent + mut_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace worst solution in current population
                # Use a simple criterion: replace the solution with the worst sum of normalized objectives
                # Find dominated solutions to replace
                worst_idx = self._find_worst_index(F, fc)
                X[worst_idx] = child
                F[worst_idx] = fc

        return X

    def _non_dominated_mask(self, F):
        """Return boolean mask of non-dominated solutions."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if F[j] <= F[i] in all objectives and < in at least one
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    def _find_worst_index(self, F, new_f):
        """Find the index to replace: prefer dominated solutions, then worst by sum."""
        n = len(F)
        # First check if new solution is dominated by any existing
        # If so, don't replace (but we must return something, so find worst)
        
        # Find all dominated solutions
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        if np.any(dominated):
            # Replace the dominated solution with the worst sum
            dominated_indices = np.where(dominated)[0]
            sums = np.sum(F[dominated_indices], axis=1)
            worst_local = np.argmax(sums)
            return dominated_indices[worst_local]
        
        # No dominated solutions: replace the one with the worst sum
        sums = np.sum(F, axis=1)
        return int(np.argmax(sums))
# EVOLVE-BLOCK-END
