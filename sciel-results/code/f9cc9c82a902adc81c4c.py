"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 10
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fpop = np.array([func(x) for x in pop])
        budget_used = pop_size
        
        # Track best
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]
        
        # Main loop
        while budget_used < budget:
            if time.time() > deadline:
                break
            
            # Generate mutant for each individual
            for i in range(pop_size):
                if budget_used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select 3 distinct random individuals different from i
                while True:
                    r1, r2, r3 = rng.integers(0, pop_size, 3)
                    if r1 != r2 and r2 != r3 and r1 != r3 and r1 != i and r2 != i and r3 != i:
                        break
                
                # DE/rand/1 mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Binomial crossover
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                budget_used += 1
                
                # Selection
                if f_trial < fpop[i]:
                    pop[i] = trial
                    fpop[i] = f_trial
                    
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        best_idx = i
                        # Optional: Shrink F slightly as we get closer? 
                        # For Bent Cigar, keeping F high helps escape narrow valleys.
                        # But we can slightly reduce it to refine.
                        # F = max(0.5, F * 0.99)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
