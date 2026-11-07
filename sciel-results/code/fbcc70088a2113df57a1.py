"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """Differential Evolution for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        t0 = time.time()
        deadline = t0 + 285.0

        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # Differential Evolution parameters
        pop_size = 20  # 20 individuals per generation
        F = 0.8        # Mutation factor
        CR = 0.5       # Crossover rate
        max_generations = budget // pop_size  # ~25 generations

        # Initialize population uniformly in [-5, 5]^d
        population = rng.uniform(-5, 5, (pop_size, d))
        fitnesses = np.zeros(pop_size)
        
        for i in range(pop_size):
            if calls >= budget or time.time() > deadline:
                return population[np.argmin(fitnesses)], fitnesses[np.argmin(fitnesses)]
            fitnesses[i] = float(func(population[i]))
            calls += 1

        # Find initial best
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]

        # Main DE loop
        for gen in range(max_generations):
            if calls >= budget or time.time() > deadline:
                break
            
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = population[r1] + F * (population[r2] - population[r3])
                v = clip(v)
                
                # Crossover: binomial crossover between v and x[i]
                trial = population[i].copy()
                j_rand = rng.integers(0, d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                trial = clip(trial)
                
                # Evaluate trial
                trial_f = float(func(trial))
                calls += 1
                
                # Selection
                if trial_f < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                
                # Safety check
                if calls >= budget or time.time() > deadline:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
