"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution Optimizer for BBOB f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Differential Evolution Parameters
        NP = 12  # Population size
        F = 0.8   # Scale factor
        CR = 0.9  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, size=(NP, dim))
        
        # Evaluate initial population
        f_pop = np.zeros(NP)
        calls_used = 0
        for i in range(NP):
            f_pop[i] = func(pop[i])
            calls_used += 1
            
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]
        
        remaining = budget - calls_used
        
        # Main loop
        while remaining >= NP:
            new_pop = pop.copy()
            new_f_pop = f_pop.copy()
            
            for i in range(NP):
                if remaining < 1:
                    break
                
                # Select 3 distinct random indices different from i
                candidates = np.delete(np.arange(NP), i)
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                
                mask = rng.random(dim) < CR
                mask[j_rand] = True # Ensure at least one gene comes from v
                
                trial[mask] = v[mask]
                
                # Evaluate trial
                f_trial = func(trial)
                calls_used += 1
                remaining -= 1
                
                # Selection
                if f_trial <= f_pop[i]:
                    new_pop[i] = trial
                    new_f_pop[i] = f_trial
                    
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            pop = new_pop
            f_pop = new_f_pop
            
        return best_x, best_f
# EVOLVE-BLOCK-END
