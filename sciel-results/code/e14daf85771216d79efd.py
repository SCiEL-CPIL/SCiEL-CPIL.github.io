"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive mutation for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial population of 50 individuals
        pop_size = min(50, budget)
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        fitness = np.array([func(x) for x in pop])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        calls_made = pop_size
        F = 0.8  # mutation factor
        CR = 0.9  # crossover rate

        # Track convergence for adaptive F
        prev_best = best_f
        stagnant = 0

        while calls_made < budget:
            # Time guard
            if time.time() > deadline:
                break

            # Adaptive F: increase when stagnant, decrease when improving
            if best_f < prev_best - 1e-6:
                stagnant = 0
                F = max(0.5, F * 0.9)
            else:
                stagnant += 1
                F = min(1.0, F * 1.05)
            prev_best = best_f

            # Generate offspring for each individual in population
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover: binomial
                r = rng.random()
                j_rand = rng.integers(0, dim)
                trial = np.where(
                    rng.random(dim) < CR or np.arange(dim) == j_rand,
                    v,
                    pop[i]
                )
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                calls_made += 1

                # Selection
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
