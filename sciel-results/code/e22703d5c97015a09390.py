"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution (DE/rand/1/bin) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, leave 10s
        
        dim = self.dim
        budget = self.budget
        
        # Parameters
        N_p = 20  # Population size
        F = 0.7   # Mutation factor
        CR = 0.5  # Crossover rate
        
        if budget < N_p:
             # Fallback to random search if budget is too small for DE
             rng = np.random.default_rng(0)
             best_x = rng.uniform(-5, 5, dim)
             best_f = func(best_x)
             for _ in range(budget - 1):
                 x = rng.uniform(-5, 5, dim)
                 f = func(x)
                 if f < best_f:
                     best_x, best_f = x, f
             return best_x, best_f

        rng = np.random.default_rng(0)
        
        # Initialize population
        pop = rng.uniform(-5, 5, (N_p, dim))
        fvals = np.array([func(x) for x in pop])
        budget -= N_p
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx]
        best_f = fvals[best_idx]
        
        while budget >= N_p and time.time() < deadline:
            # Generate trial vectors
            new_fvals = fvals.copy()
            new_pop = pop.copy()
            
            for i in range(N_p):
                if budget <= 0: break
                if time.time() > deadline: break
                
                # Select two distinct random indices different from i
                r1, r2 = rng.choice(N_p, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(N_p, 2, replace=False)
                
                # Mutation: v = pop[r1] + F * (pop[r2] - pop[i])
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover: binomial
                j_rand = rng.integers(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Evaluation
                f_trial = func(trial)
                budget -= 1
                
                # Selection
                if f_trial < fvals[i]:
                    new_pop[i] = trial
                    new_fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial
            
            pop = new_pop
            fvals = new_fvals
            
            # Check for stagnation or convergence? 
            # DE usually handles this well. 
            # We can stop if no improvement in last few generations to save time? 
            # But budget is small (500), so let's use it all if time permits.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
