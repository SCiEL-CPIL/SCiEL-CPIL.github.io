"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with adaptive step-size and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population: stratified sampling for better coverage
        pop_size = 30
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            X[:, d] = np.linspace(lo[d], hi[d], pop_size)
        # Add some random points for diversity
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive step size
        base_step = 0.15

        while used < self.budget:
            # Check time deadline
            if time.time() > deadline:
                break

            # Find non-dominated solutions
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            non_dom_idx = np.where(~dominated)[0]

            if len(non_dom_idx) == 0:
                non_dom_idx = np.array([0])

            # Select parent from non-dominated set
            parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            parent = X[parent_idx]

            # Adaptive step size: smaller steps for refinement
            step_scale = base_step * (0.5 + 0.5 * (used / self.budget))
            child = parent + step_scale * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst dominated solution or random if all non-dominated
            if np.any(dominated):
                worst_idx = np.where(dominated)[0]
                # Pick the worst among dominated (highest sum of objectives)
                worst_local = worst_idx[np.argmax(np.sum(F[worst_idx], axis=1))]
            else:
                worst_local = rng.integers(len(F))

            X[worst_local], F[worst_local] = child, fc

            # Occasionally do a local refinement pass on best non-dominated point
            if used % 20 == 0 and used < self.budget - 10:
                best_nd = non_dom_idx[np.argmin(np.sum(F[non_dom_idx], axis=1))]
                for _ in range(min(5, self.budget - used)):
                    if time.time() > deadline:
                        break
                    fine_child = X[best_nd] + 0.02 * span * rng.standard_normal(self.dim)
                    fine_child = np.clip(fine_child, lo, hi)
                    fine_fc = problem(fine_child)
                    used += 1
                    # Only replace if it's non-dominated or improves
                    if np.all(fine_fc <= F[best_nd]) and np.any(fine_fc < F[best_nd]):
                        X[best_nd], F[best_nd] = fine_child, fine_fc
                    elif used >= self.budget:
                        break

        return X
# EVOLVE-BLOCK-END
