"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard and global best tracking."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        
        # DE parameters
        pop_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        population = np.random.uniform(-5, 5, (pop_size, dim))
        
        # Evaluate initial population
        fitness = np.empty(pop_size)
        calls_made = 0
        best_x = None
        best_fx = np.inf

        for i in range(pop_size):
            if time.time() >= deadline:
                break
            f_val = func(population[i])
            fitness[i] = f_val
            calls_made += 1
            if f_val < best_fx:
                best_fx = f_val
                best_x = population[i].copy()

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls_made += 1

        # Main loop
        while calls_made < budget and time.time() < deadline:
            # For each individual in the population, try to improve it
            for i in range(pop_size):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Select 3 distinct random indices different from i
                # We need 3 distinct indices from the population
                # Ensure they are not i
                candidates = [j for j in range(pop_size) if j != i]
                # Shuffle and pick 3
                np.random.shuffle(candidates)
                idx1, idx2, idx3 = candidates[0], candidates[1], candidates[2]
                
                # Generate mutant vector
                # v = x[idx1] + F * (x[idx2] - x[idx3])
                mutant = population[idx1] + F * (population[idx2] - population[idx3])
                
                # Clip to bounds
                mutant = np.clip(mutant, -5, 5)
                
                # Evaluate mutant
                f_mutant = func(mutant)
                calls_made += 1
                
                # Update global best if needed
                if f_mutant < best_fx:
                    best_fx = f_mutant
                    best_x = mutant.copy()
                
                # Tournament: replace if mutant is better than current individual
                if f_mutant < fitness[i]:
                    population[i] = mutant
                    fitness[i] = f_mutant

        return best_x, best_fx
# EVOLVE-BLOCK-END
