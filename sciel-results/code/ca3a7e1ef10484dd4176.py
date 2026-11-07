"""Rewritten optimizer: multi-restart CMA-ES with time guard and final polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma and final local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # 10s safety margin

        # Global best tracking
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64).clip(-5, 5)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Differential Evolution Parameters
        pop_size = 20
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate

        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([evaluate(population[i]) for i in range(pop_size)])
        
        # Update best from initial population
        idx_best = np.argmin(fitness)
        best_x = population[idx_best].copy()
        best_f = fitness[idx_best]

        # DE Loop
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if call_count + pop_size > budget:
                break
                
            # Generate offspring
            for i in range(pop_size):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                # DE/rand/1 mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                j_rand = rng.integers(dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                trial = np.clip(trial, -5, 5)
                f_trial = evaluate(trial)
                
                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # Adaptive parameters
            F = np.clip(F * 0.99, 0.4, 1.0)
            CR = np.clip(CR + 0.01, 0.1, 0.9)

        # Final Polish with Nelder-Mead if budget remains
        if call_count < budget and not np.isnan(best_x).any():
            x_curr = best_x.copy()
            f_curr = best_f
            
            # Simple coordinate descent / Nelder-Mead hybrid for final steps
            # We just do a few local steps if we have time
            step = 0.5
            for _ in range(10):
                if call_count >= budget or time.time() > deadline:
                    break
                for i in range(dim):
                    if call_count >= budget:
                        break
                    for direction in [-1, 1]:
                        x_try = x_curr.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = evaluate(x_try)
                        if f_try < f_curr:
                            x_curr = x_try
                            f_curr = f_try
                            if f_curr < best_f:
                                best_f = f_curr
                                best_x = x_curr.copy()
                step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
