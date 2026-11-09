"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based differential evolution with fixed budget."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 20
        calls_made = 0
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in pop])
        calls_made += pop_size
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Differential evolution parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        
        # Remaining budget
        remaining = budget - calls_made
        generations = remaining // pop_size
        
        for gen in range(generations):
            # Generate trial vectors
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                    
                # Select three distinct random indices
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                # Differential mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Binary crossover
                trial = np.where(rng.random(dim) < CR, v, pop[i])
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                trial_f = func(trial)
                calls_made += 1
                
                # Selection
                if trial_f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
        
        # Use any remaining budget for local refinement around best
        while calls_made < budget:
            # Simple local search around best
            step = max(0.1, 5.0 / (budget - calls_made + 1))
            candidate = best_x + step * rng.standard_normal(dim)
            candidate = np.clip(candidate, -5, 5)
            cand_f = func(candidate)
            calls_made += 1
            if cand_f < best_f:
                best_f = cand_f
                best_x = candidate.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
