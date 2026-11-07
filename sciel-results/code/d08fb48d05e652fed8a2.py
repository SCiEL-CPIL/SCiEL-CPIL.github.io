"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer with adaptive parameters."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            return f

        # Population size
        N = 15
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate

        # Initialize population
        population = rng.uniform(-5.0, 5.0, (N, dim))
        fitness = np.array([safe_call(ind) for ind in population])
        
        # Track best
        idx_best = np.argmin(fitness)
        best_f = fitness[idx_best]
        best_x = population[idx_best].copy()

        # Differential Evolution Loop
        while calls_made < budget:
            # Check if we can afford a full generation
            if calls_made + N > budget:
                # Evaluate only as many as possible
                N_this = budget - calls_made
            else:
                N_this = N

            new_population = population.copy()
            new_fitness = fitness.copy()

            for i in range(N_this):
                # Select three distinct random indices different from i
                # Efficiently generate 3 unique indices
                r1, r2, r3 = rng.choice(N, size=3, replace=False)
                while r1 == i:
                    r1 = rng.integers(0, N)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, N)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(0, N)

                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = population[r1] + F * (population[r2] - population[r3])

                # Crossover
                trial = population[i].copy()
                # Choose random dimension to force inclusion of v
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluation
                f_trial = safe_call(trial)
                
                # Selection
                if f_trial <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

            population = new_population
            fitness = new_fitness

            # Simple convergence check: if best hasn't improved in many generations, break
            # We don't track generations explicitly, but we can check if variance is small
            if N_this < N:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
