import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin
        
        d = self.dim
        N = self.budget
        
        # Differential Evolution Parameters
        NP = 10  # Population size
        F = 0.8  # Scaling factor
        CR = 0.5 # Crossover rate
        
        # Initialize population
        # Use a seed for reproducibility in local tests, but random is fine for evaluator
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (NP, d))
        
        # Evaluate initial population
        fit = np.empty(NP)
        calls = 0
        for i in range(NP):
            fit[i] = func(pop[i])
            calls += 1
            if calls >= N:
                break
                
        best_idx = np.argmin(fit[:calls])
        best_x = pop[best_idx]
        best_f = fit[best_idx]
        
        # Main loop
        while calls < N:
            # Time check
            if time.time() - start_time > timeout:
                break
                
            # Iterate through population
            for i in range(NP):
                if calls >= N:
                    break
                    
                # Select 3 distinct indices different from i
                # Ensure we have at least 4 individuals if NP > 3
                if NP <= 3:
                    break
                candidates = [k for k in range(NP) if k != i]
                j1, j2, j3 = rng.choice(candidates, 3, replace=False)
                
                # Generate trial vector using DE/rand/1/bin
                v = pop[j1] + F * (pop[j2] - pop[j3])
                
                # Binomial crossover
                r = rng.random(d)
                j_rand = rng.integers(d)
                trial = np.where(r < CR, v, pop[i])
                trial[j_rand] = v[j_rand]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial
                        
        return best_x, best_f
