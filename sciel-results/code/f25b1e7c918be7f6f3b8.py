"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: LHS init + population-based search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Population size: balance between diversity and exploitation
        pop_size = min(20, max(10, budget // 10))

        # 1. Initialization: Latin Hypercube Sampling for diverse start
        n_init = pop_size

        # Generate LHS points
        perm = np.argsort(np.argsort(np.random.rand(n_init, dim)), axis=0)
        u = (perm + np.random.rand(n_init, dim)) / n_init
        X = lo + (hi - lo) * u

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_init

        # 2. Iterative refinement
        while used < budget:
            if time.time() > deadline:
                break

            # Identify non-dominated points in population
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            # Select parent from non-dominated points, or random if all dominated
            if np.any(~dominated):
                parent_indices = np.where(~dominated)[0]
                parent_idx = parent_indices[np.random.randint(len(parent_indices))]
            else:
                parent_idx = np.random.randint(pop_size)

            x_parent = X[parent_idx]

            # Adaptive perturbation: larger early, smaller late
            progress = used / budget
            step_size = 0.15 * (1.0 - 0.5 * progress)

            # Gaussian perturbation
            noise = np.random.randn(dim) * step_size
            x_child = x_parent + noise * (hi - lo)
            x_child = np.clip(x_child, lo, hi)

            f_child = problem(x_child)
            used += 1

            if np.any(np.isnan(f_child)) or np.any(np.isinf(f_child)):
                continue

            # Replace the worst point in the population
            # Use sum of objectives as a simple fitness measure
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = x_child
            F[worst_idx] = f_child

        return X
# EVOLVE-BLOCK-END
