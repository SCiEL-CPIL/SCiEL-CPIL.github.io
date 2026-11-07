"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0

        def check_time():
            return time.time() > deadline

        rng = np.random.default_rng(42)

        # Initialize population with diverse points
        pop_size = 20
        population = np.zeros((pop_size, dim))
        fitness = np.zeros(pop_size)

        # Initialize with a mix of strategies
        for i in range(pop_size):
            if i % 3 == 0:
                population[i] = rng.uniform(-5, 5, dim)
            elif i % 3 == 1:
                population[i] = rng.uniform(-2, 2, dim)
            else:
                population[i] = rng.uniform(-3.5, 3.5, dim)
            fitness[i] = func(population[i])
            calls += 1
            if calls >= budget or check_time():
                break

        if calls >= budget or check_time():
            best_idx = np.argmin(fitness[:calls])
            return population[best_idx].copy(), fitness[best_idx]

        best_idx = np.argmin(fitness)
        best_f = fitness[best_idx]
        best_x = population[best_idx].copy()

        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.5  # Crossover rate
        stagnation = 0
        max_stagnation = 30

        while calls < budget and not check_time():
            # DE mutation and crossover
            for i in range(pop_size):
                if calls >= budget or check_time():
                    break

                # Select three distinct random indices different from i
                while True:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    if a != i and b != i and c != i:
                        break

                # DE/rand/1 mutation
                trial = population[a] + F * (population[b] - population[c])
                trial = np.clip(trial, -5, 5)

                # Binary crossover
                crossover_point = rng.randint(dim)
                mask = rng.random(dim) < CR
                mask[crossover_point] = True
                trial = np.where(mask, trial, population[i])

                # Evaluate trial
                trial_f = func(trial)
                calls += 1

                # Selection
                if trial_f < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1

                # Adaptive parameter adjustment
                if stagnation > max_stagnation:
                    # Reset stagnation and adjust parameters
                    F = np.clip(F * 1.1, 0.4, 1.0)
                    CR = np.clip(CR * 1.1, 0.2, 0.9)
                    stagnation = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
