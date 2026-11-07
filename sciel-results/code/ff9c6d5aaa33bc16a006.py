"""Optimized program for BBOB f21 (Gallagher 101 peaks) with Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Differential Evolution optimizer:
    1. Global exploration with a large population (lambda=40) and aggressive mutation (F=0.8, CR=0.9).
    2. Adaptive exploitation by reducing F and CR as the population converges.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_f = float('inf')
        best_x = None
        
        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return float('inf')
            n_calls += 1
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # DE Parameters
        # Large population for global coverage
        lambda_pop = 40
        if lambda_pop >= budget:
            lambda_pop = max(10, budget // 2)
        
        # Initialize population uniformly in [-5, 5]^dim
        population = rng.uniform(-5, 5, size=(lambda_pop, dim))
        fitness = np.full(lambda_pop, float('inf'))
        
        # Evaluate initial population
        for i in range(lambda_pop):
            if n_calls >= budget or time.time() > deadline - 5:
                break
            fitness[i] = wrapped_func(population[i])
        
        if best_x is None:
            return np.zeros(dim), 0.0

        # DE Loop
        # Start with aggressive exploration
        F_start = 0.8
        CR_start = 0.9
        F_end = 0.5
        CR_end = 0.1
        
        max_generations = budget // lambda_pop
        gen = 0
        
        while n_calls < budget and time.time() < deadline - 5 and gen < max_generations:
            gen += 1
            
            # Adaptive parameters: decrease F and CR over time to shift from exploration to exploitation
            # Linear interpolation based on generation
            if max_generations > 0:
                progress = gen / max_generations
            else:
                progress = 1.0
            
            F = F_start + (F_end - F_start) * progress
            CR = CR_start + (CR_end - CR_start) * progress
            
            # Keep a copy of current population and fitness for the next generation
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            # Sort indices by fitness to identify best, worst, etc.
            sorted_idx = np.argsort(fitness)
            best_idx = sorted_idx[0]
            worst_idx = sorted_idx[-1]
            
            # For each individual in the population, generate a trial vector
            for i in range(lambda_pop):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                
                # Select 3 distinct random indices different from i
                # To avoid expensive random sampling, we can use a simple deterministic offset or random sampling
                # Given dim=5 and lambda=40, random sampling is cheap
                candidates = [j for j in range(lambda_pop) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # DE/rand/1 mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = population[i].copy()
                # Ensure at least one dimension is from v
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
            
            population = new_population
            fitness = new_fitness

        return best_x, best_f

# EVOLVE-BLOCK-END
