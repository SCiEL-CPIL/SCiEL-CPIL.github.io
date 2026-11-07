"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = (-5, 5)

        # Population size
        pop_size = 10
        if pop_size * dim >= budget:
            # Fallback to simple random search if budget is too small for DE
            pop_size = max(2, budget // dim)

        # Initialize population
        population = rng.uniform(bounds[0], bounds[1], (pop_size, dim))
        fitness = np.array([func(x) for x in population])
        calls = pop_size

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # DE parameters
        F = 0.5  # Differential weight
        CR = 0.5  # Crossover rate

        stagnation_count = 0
        max_stagnation = 10

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Adapt F and CR based on stagnation
            if stagnation_count > max_stagnation:
                # Increase exploration
                F = min(F + 0.1, 1.0)
                CR = min(CR + 0.1, 1.0)
                # Randomly re-seed one individual
                if rng.random() < 0.5:
                    idx = rng.integers(0, pop_size)
                    population[idx] = rng.uniform(bounds[0], bounds[1], dim)
                    fitness[idx] = func(population[idx])
                    calls += 1
                    if fitness[idx] < best_f:
                        best_f = fitness[idx]
                        best_x = population[idx].copy()
                        best_idx = idx
                stagnation_count = 0

            improved = False
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break

                # Select 3 distinct random individuals
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)

                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, bounds[0], bounds[1])

                # Crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluation
                f_trial = func(trial)
                calls += 1

                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    improved = True

                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        best_idx = i

            if not improved:
                stagnation_count += 1
            else:
                stagnation_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
