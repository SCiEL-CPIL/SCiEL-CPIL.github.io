import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 290.0 # 10 seconds safety margin
        
        # Parameters
        lambda_pop = 5
        F = 0.8
        CR = 0.5
        
        # Initialize population
        # Uniform in [-5, 5]
        pop = rng.uniform(-5, 5, (lambda_pop, self.dim))
        
        # Evaluate initial population
        fitness = np.zeros(lambda_pop)
        calls = 0
        for i in range(lambda_pop):
            if calls >= self.budget or time.time() > deadline:
                break
            fitness[i] = func(pop[i])
            calls += 1
        
        if calls == 0:
            # Should not happen, but handle gracefully
            return np.zeros(self.dim), 0.0

        # Find initial best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # Main Loop
        # We have budget - calls remaining.
        # Each generation uses lambda_pop evaluations.
        while calls + lambda_pop <= self.budget and time.time() < deadline:
            
            # Generate trial vectors
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            # Update best within this generation? 
            # Standard DE usually uses the best from the *previous* generation for mutation.
            # Using the current best is "Best/1" with online update. 
            # Let's stick to using the best from the start of the generation to avoid bias in the same pass.
            gen_best_x = best_x
            gen_best_f = best_f
            
            for i in range(lambda_pop):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                # Select r1, r2 distinct from i and each other
                # Available indices: 0..4 excluding i
                candidates = [j for j in range(lambda_pop) if j != i]
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                
                # Mutation: v = x_best + F * (x_r1 - x_r2)
                v = gen_best_x + F * (pop[r1] - pop[r2])
                
                # Clip to box
                v = np.clip(v, -5, 5)
                
                # Crossover
                u = pop[i].copy()
                # Random integer in [0, dim-1] to ensure at least one gene is from v
                j_rand = rng.integers(0, self.dim)
                
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Evaluate
                f_u = func(u)
                calls += 1
                
                # Selection
                if f_u < fitness[i]:
                    pop[i] = u
                    fitness[i] = f_u
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    
                    # Update global best
                    if f_u < best_f:
                        best_x = u.copy()
                        best_f = f_u
                        # Optionally update gen_best_x for subsequent mutations in this generation?
                        # Using online best can accelerate convergence but might reduce diversity.
                        # Let's update it to leverage the "attractive" nature quickly.
                        gen_best_x = best_x
                        gen_best_f = best_f

            # End of generation
            # Update pop and fitness to new versions (already done in-place above)
            # No need to explicitly assign new_pop if we updated pop in place
            
            # Check for stagnation? 
            # If best_f hasn't improved for many generations, we might need to restart or perturb.
            # With only ~100 generations, simple DE might be enough.
            
        return best_x, best_f
