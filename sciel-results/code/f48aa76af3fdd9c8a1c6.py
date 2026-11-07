"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with non-dominated front tracking."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Initial population
        pop_size = min(30, self.budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track non-dominated front indices
        def get_non_dominated(F):
            n = len(F)
            is_nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nd[i]:
                    continue
                for j in range(n):
                    if i == j or not is_nd[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_nd[i] = False
                        break
            return np.where(is_nd)[0]

        # Main loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Get non-dominated front
            nd_idx = get_non_dominated(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Select parent from non-dominated front
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X[parent_idx]

            # Adaptive mutation: larger steps early, smaller later
            progress = used / self.budget
            step_size = (hi - lo) * (0.3 * (1 - progress) + 0.05)
            child = parent + step_size * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst point in the non-dominated front
            # Use sum of objectives as a simple quality measure
            nd_F = F[nd_idx]
            worst_in_nd = nd_idx[np.argmax(np.sum(nd_F, axis=1))]

            # Only replace if child is not dominated by the worst
            # Simple check: if child has lower sum, replace
            if np.sum(fc) < np.sum(F[worst_in_nd]):
                X[worst_in_nd] = child
                F[worst_in_nd] = fc
            elif rng.random() < 0.3:
                # Occasionally replace anyway for diversity
                X[worst_in_nd] = child
                F[worst_in_nd] = fc

        return X
# EVOLVE-BLOCK-END
