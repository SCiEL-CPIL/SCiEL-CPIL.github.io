# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """JADE-style Differential Evolution for BBOB f18 with wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = 280.0
        dim = self.dim
        budget = self.budget
        L, U = -5.0, 5.0

        rng = np.random.default_rng(20240517)

        # Population size
        NP = 10
        F = 0.8
        CR = 0.1

        # Initialize population
        pop = rng.uniform(L, U, (NP, dim))
        fitness = np.array([float(func(x)) for x in pop])
        n_calls = NP

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # JADE parameters
        F1 = 0.5
        F2 = 0.5
        CR1 = 0.1
        CR2 = 0.9

        # Track for JADE adaptation
        F_list = []
        CR_list = []

        while n_calls < budget:
            if time.time() - t0 > DEADLINE:
                break

            # Sort by fitness
            order = np.argsort(fitness)
            pop_sorted = pop[order]
            fit_sorted = fitness[order]

            # Select best individual
            x_best = pop_sorted[0]

            for i in range(NP):
                if n_calls >= budget or time.time() - t0 > DEADLINE:
                    break

                # JADE mutation strategy: "DE/best/1" and "DE/rand/1"
                if rng.random() < 0.5:
                    # DE/best/1
                    p1_idx = rng.integers(1, NP)
                    p2_idx = rng.integers(1, NP)
                    while p2_idx == p1_idx:
                        p2_idx = rng.integers(1, NP)
                    v = x_best + F * (pop_sorted[p1_idx] - pop_sorted[p2_idx])
                else:
                    # DE/rand/1
                    p1_idx = rng.integers(0, NP)
                    p2_idx = rng.integers(0, NP)
                    p3_idx = rng.integers(0, NP)
                    while p2_idx == p1_idx or p3_idx == p1_idx:
                        p2_idx = rng.integers(0, NP)
                        p3_idx = rng.integers(0, NP)
                    v = pop_sorted[p1_idx] + F * (pop_sorted[p2_idx] - pop_sorted[p3_idx])

                # Clip to bounds
                v = np.clip(v, L, U)

                # Binomial crossover
                trial = pop_sorted[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                ft = float(func(trial))
                n_calls += 1

                # Selection
                if ft < fit_sorted[i]:
                    pop_sorted[i] = trial
                    fit_sorted[i] = ft
                    F_list.append(F)
                    CR_list.append(CR)

                    # Update best
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()

            # Unsort back to original order (not strictly necessary, but keep consistent)
            # Actually, we can just work with sorted population for next iteration
            pop = pop_sorted
            fitness = fit_sorted

            # JADE adaptation of F and CR
            if F_list:
                F = F1 * rng.standard_normal() * 0.3 + F1
                F = np.clip(F, 0.1, 0.9)
                CR = CR1 * rng.standard_normal() * 0.3 + CR1
                CR = np.clip(CR, 0.0, 1.0)
                F_list = []
                CR_list = []

        return best_x
# EVOLVE-BLOCK-END
