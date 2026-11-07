"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution with adaptive parameters for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deterministic seed for reproducibility
        seed = 12345
        rng = np.random.default_rng(seed)

        # Population size
        NP = 25
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate

        # Initialize population
        pop = np.zeros((NP, dim))
        fits = np.zeros(NP)

        # Initialize with diverse points
        pop[0] = np.zeros(dim)  # Center
        for i in range(1, NP):
            pop[i] = rng.uniform(-5, 5, dim)

        # Evaluate initial population
        for i in range(NP):
            fits[i] = func(pop[i])

        calls_used = NP

        # Track best
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]

        # Adaptive parameters
        generation = 0
        max_generations = (budget - NP) // NP + 10

        while calls_used < budget and generation < max_generations:
            generation += 1
            new_pop = np.zeros((NP, dim))
            new_fits = np.zeros(NP)

            # Copy current population
            for i in range(NP):
                new_pop[i] = pop[i]
                new_fits[i] = fits[i]

            # For each individual, create a trial vector
            for i in range(NP):
                if calls_used >= budget:
                    break

                # Select three distinct random individuals
                while True:
                    r1, r2, r3 = rng.choice(NP, 3, replace=False)
                    if r1 != r2 and r2 != r3 and r1 != r3 and r1 != i and r2 != i and r3 != i:
                        break

                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                calls_used += 1

                # Selection
                if f_trial < fits[i]:
                    new_pop[i] = trial
                    new_fits[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

            # Update population
            pop = new_pop
            fits = new_fits

            # Adaptive mutation factor: decrease F over time for convergence
            F = max(0.3, 0.8 - 0.005 * generation)

        # Use remaining budget for local refinement
        remaining = budget - calls_used
        if remaining > 0:
            sigma = 0.5
            for _ in range(remaining):
                step = rng.normal(0, sigma, dim)
                x_trial = np.clip(best_x + step, -5, 5)
                f_trial = func(x_trial)
                calls_used += 1

                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                sigma = max(sigma, 1e-8)

        return best_x, best_f
# EVOLVE-BLOCK-END
