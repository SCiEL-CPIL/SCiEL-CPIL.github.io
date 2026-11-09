# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (DE) with binomial crossover."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            if calls_made[0] >= budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made[0] += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # DE Parameters
        pop_size = 10
        F = 0.8
        CR = 0.5
        
        # Initialize population
        # Start with center and random points
        pop = np.zeros((pop_size, n))
        pop[0] = 0.0
        pop[1:] = rng.uniform(-4, 4, (pop_size - 1, n))
        
        fit = np.zeros(pop_size)
        for i in range(pop_size):
            fit[i] = wrapped_func(pop[i])
        
        # Track best in population
        best_idx = np.argmin(fit)
        current_best_f = fit[best_idx]
        current_best_x = pop[best_idx].copy()

        # Main loop
        while calls_made[0] < budget:
            if time.time() > deadline:
                break
            
            # Update global best
            if current_best_f < best_f:
                best_f = current_best_f
                best_x = current_best_x.copy()

            for i in range(pop_size):
                if calls_made[0] >= budget:
                    break
                
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover (Binomial)
                trial = pop[i].copy()
                j_rand = rng.integers(n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    if f_trial < current_best_f:
                        current_best_f = f_trial
                        current_best_x = trial.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
