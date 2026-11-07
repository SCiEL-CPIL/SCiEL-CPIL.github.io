"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution multi-objective search for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        # 330s kill limit, use 325s as deadline for safety
        deadline = t_start + 325.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        budget = self.budget

        # Initial population size
        pop_size = 100
        if pop_size > budget:
            pop_size = budget

        # Use standard numpy random (seeded by evaluator)
        X = lo + (hi - lo) * np.random.random((pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Differential Evolution parameters
        F_factor = 0.8
        CR = 0.9

        while used < budget:
            if time.time() > deadline:
                break

            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select 3 distinct random parents
                idx = np.random.choice(pop_size, 3, replace=False)
                x1, x2, x3 = X[idx[0]], X[idx[1]], X[idx[2]]
                xi = X[i]

                # Differential evolution mutation
                v = x1 + F_factor * (x2 - x3)

                # Crossover
                j_rand = np.random.randint(n_dim)
                mask = np.random.random(n_dim) < CR
                mask[j_rand] = True

                trial = np.where(mask, v, xi)
                trial = np.clip(trial, lo, hi)

                f_trial = problem(trial)
                used += 1

                # Selection
                if np.all(f_trial <= F[i]) and np.any(f_trial < F[i]):
                    X[i] = trial
                    F[i] = f_trial

        return X
# EVOLVE-BLOCK-END
