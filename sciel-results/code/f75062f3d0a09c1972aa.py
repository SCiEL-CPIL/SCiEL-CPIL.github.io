"""Multi-objective optimizer for ZDT1 using a decomposition-based approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective optimizer for ZDT1 (2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # For ZDT1: f1 = x[0], f2 = (1 + 10/(n-1) * sum(x[1:])) * (1 - sqrt(f1/f1_max))
        # The Pareto front is f2 = 1 - sqrt(f1) for f1 in [0, 1]
        # We want to distribute points along this front to maximize hypervolume
        # with reference point [11, 11].

        # Strategy: Use a population-based search that targets the Pareto front.
        # Key insight: ZDT1's Pareto front is known analytically.
        # x[0] controls f1, x[1:] should be 0 to minimize f2 for a given f1.
        # So optimal solutions have x[1:] = 0 and x[0] in [0, 1].

        # We'll use a hybrid approach:
        # 1. Sample points along the known Pareto front structure
        # 2. Use evolutionary search to refine

        pop_size = 50
        X = np.zeros((pop_size, self.dim))
        
        # Initialize: spread x[0] across [0, 1], set x[1:] = 0
        # This directly targets the Pareto front
        X[:, 0] = np.linspace(0.01, 0.99, pop_size)
        
        # Add some random exploration
        for i in range(5):
            X[i] = lo + (hi - lo) * rng.random(self.dim)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Check if we're within budget and time
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Find non-dominated solutions
            non_dom_mask = self._nondominated_mask(F)
            
            if np.sum(non_dom_mask) > 0:
                parents = X[non_dom_mask]
                F_parents = F[non_dom_mask]
            else:
                parents = X
                F_parents = F

            # Select parents and create children
            n_children = min(pop_size, self.budget - used)
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                
                # Crossover: blend two parents
                i1 = rng.integers(len(parents))
                i2 = rng.integers(len(parents))
                
                if i1 == i2:
                    child = parents[i1].copy()
                else:
                    alpha = rng.random(self.dim)
                    child = alpha * parents[i1] + (1 - alpha) * parents[i2]
                
                # Mutation
                # For ZDT1, key insight: x[1:] should be near 0
                # Apply targeted mutation
                if rng.random() < 0.7:
                    # Push x[1:] toward 0
                    child[1:] = child[1:] * rng.random(self.dim - 1)
                else:
                    # Random mutation
                    child += 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                # Replace worst individual
                # Use a simple fitness: sum of normalized objectives
                # Points closer to (0,0) are better
                worst_idx = self._find_worst_index(F)
                X[worst_idx] = child
                F[worst_idx] = fc

        return X

    def _nondominated_mask(self, F):
        """Return boolean mask of non-dominated solutions."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j <= i in all objectives and j < i in at least one
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    def _find_worst_index(self, F):
        """Find the index of the worst individual based on simple sum."""
        # Normalize and sum
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1
        F_norm = (F - f_min) / f_range
        return int(np.argmax(np.sum(F_norm, axis=1)))
# EVOLVE-BLOCK-END
