"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)
        pop_size = 20

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Differential Evolution loop
        F_scale = 0.8
        CR = 0.9

        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            for i in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break

                # Select 3 distinct random indices different from i
                while True:
                    j = rng.integers(pop_size)
                    if j != i:
                        break
                while True:
                    k = rng.integers(pop_size)
                    if k != i and k != j:
                        break
                while True:
                    l = rng.integers(pop_size)
                    if l != i and l != j and l != k:
                        break

                # DE/rand/1 mutation
                v = X[j] + F_scale * (X[k] - X[l])
                v = np.clip(v, lo, hi)

                # Binomial crossover
                r = rng.random(dim)
                jrand = rng.integers(dim)
                u = np.where(r <= CR, v, X[i])
                u[jrand] = v[jrand]

                # Evaluate
                fu = problem(u)
                used += 1

                # Selection: keep better of u and X[i]
                if np.all(fu <= F[i]):
                    if np.any(fu < F[i]):
                        X[i] = u
                        F[i] = fu
                else:
                    # Check if X[i] dominates u
                    if not (np.all(F[i] <= fu) and np.any(F[i] < fu)):
                        # Incomparable: keep current (could use more sophisticated strategy)
                        pass

        return X
# EVOLVE-BLOCK-END
