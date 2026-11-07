"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid Differential Evolution with Local Population Refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Phase 1: Differential Evolution for global exploration
        # We allocate 50% of the budget to DE.
        de_budget = int(self.budget * 0.5)
        pop_size = 20
        # Ensure pop_size is not larger than de_budget
        if pop_size > de_budget:
            pop_size = max(2, de_budget)
            
        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, self.dim))
        fitnesses = np.array([func(ind) for ind in population])
        
        # DE Parameters
        F = 0.8
        CR = 0.9
        
        # Run DE for remaining DE budget
        calls_made = pop_size
        while calls_made < de_budget:
            if calls_made + pop_size > de_budget:
                break
                
            # Update population
            new_fitnesses = fitnesses.copy()
            for i in range(pop_size):
                # Select two distinct random indices
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(pop_size, 2, replace=False)
                
                # Create trial vector
                v = population[i] + F * (population[r1] - population[r2])
                v = np.clip(v, -5, 5)
                
                fv = func(v)
                calls_made += 1
                
                # Selection
                if fv < fitnesses[i]:
                    population[i] = v
                    new_fitnesses[i] = fv
            
            fitnesses = new_fitnesses

        # Find the best individual from DE phase
        best_idx = np.argmin(fitnesses)
        x_best = population[best_idx].copy()
        f_best = fitnesses[best_idx]
        
        # Phase 2: Local Population-Based Refinement (Small DE)
        # This is more robust than (1+1)-ES for Weierstrass as it maintains diversity
        remaining_budget = self.budget - calls_made
        if remaining_budget > 0:
            local_pop_size = 5
            if local_pop_size > remaining_budget:
                local_pop_size = max(2, remaining_budget)
            
            # Initialize local population around the best point
            # Use a small initial spread
            spread = 0.5
            local_population = x_best + rng.uniform(-spread, spread, (local_pop_size, self.dim))
            local_population = np.clip(local_population, -5, 5)
            local_fitnesses = np.array([func(ind) for ind in local_population])
            calls_made += local_pop_size
            
            # Local DE Parameters
            local_F = 0.5
            local_CR = 0.5
            
            while calls_made < self.budget:
                if calls_made + local_pop_size > self.budget:
                    break
                    
                new_local_fitnesses = local_fitnesses.copy()
                for i in range(local_pop_size):
                    # Select two distinct random indices
                    r1, r2 = rng.choice(local_pop_size, 2, replace=False)
                    while r1 == i or r2 == i:
                        r1, r2 = rng.choice(local_pop_size, 2, replace=False)
                    
                    # Create trial vector
                    v = local_population[i] + local_F * (local_population[r1] - local_population[r2])
                    v = np.clip(v, -5, 5)
                    
                    fv = func(v)
                    calls_made += 1
                    
                    # Selection
                    if fv < local_fitnesses[i]:
                        local_population[i] = v
                        new_local_fitnesses[i] = fv
                
                local_fitnesses = new_local_fitnesses
                
                # Update best
                local_best_idx = np.argmin(local_fitnesses)
                if local_fitnesses[local_best_idx] < f_best:
                    f_best = local_fitnesses[local_best_idx]
                    x_best = local_population[local_best_idx].copy()
                    
                # Optionally shrink the population around the best if progress is slow
                # But for simplicity, we just run the small DE

        return x_best, f_best
# EVOLVE-BLOCK-END
