"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # N = 10 * dim is standard, but for small dim and limited budget, 
        # N=10 (absolute) or N=5*dim might be better to allow more generations.
        # Let's use N = max(10, 5 * dim)
        N = max(10, 5 * dim)
        
        # Ensure we don't exceed budget in initialization
        if N >= budget:
            # Fallback to random search if budget is too small for DE
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x
            return best_x, best_f

        F = 0.8
        CR = 0.9
        
        # Initialize population
        population = rng.uniform(-5, 5, (N, dim))
        fitness = np.array([func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]
        
        count = N
        
        # Main loop
        while count + N <= budget:
            # Generate offspring
            new_population = []
            new_fitness = []
            
            for i in range(N):
                # Mutation
                r1, r2 = rng.choice(N, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(N, 2, replace=False)
                    
                mutant = population[best_idx] + F * (population[r1] - population[r2])
                
                # Crossover
                cross_point = rng.integers(0, dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == cross_point:
                        trial[j] = mutant[j]
                        
                # Boundary handling (reflection)
                for j in range(dim):
                    if trial[j] < -5:
                        trial[j] = -5 + (-5 - trial[j]) % 10
                    elif trial[j] > 5:
                        trial[j] = 5 - (trial[j] - 5) % 10
                        
                # Evaluate
                f_trial = func(trial)
                count += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(f_trial)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
                    
            population = np.array(new_population)
            fitness = np.array(new_fitness)
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_f:
                best_f = fitness[best_idx]
                best_x = population[best_idx].copy()
                
        # Use remaining budget for local random search
        if count < budget:
            sigma = 0.5
            for _ in range(budget - count):
                x = best_x + rng.normal(0, sigma, dim)
                x = np.clip(x, -5, 5)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
