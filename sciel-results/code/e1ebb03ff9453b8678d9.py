import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # Initialize
        rng = np.random.default_rng(42)
        
        # Population size for DE
        pop_size = 10
        if pop_size < 4:
            pop_size = 4
            
        # Initialize population uniformly in [-5, 5]
        population = rng.uniform(-5, 5, (pop_size, n))
        
        # Evaluate initial population
        fitness = np.zeros(pop_size)
        calls_made = 0
        best_x = population[0].copy()
        best_f = np.inf
        
        for i in range(pop_size):
            if calls_made >= budget:
                break
            f_val = func(population[i])
            fitness[i] = f_val
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = population[i].copy()
                
        # DE Parameters
        F = 0.8
        CR = 0.5
        
        # Phase 1: Differential Evolution
        # We will run DE until we have used about 70% of the budget
        de_budget_limit = int(budget * 0.7)
        
        while calls_made < de_budget_limit:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate trial vectors
            for i in range(pop_size):
                if calls_made >= de_budget_limit:
                    break
                    
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = np.zeros(n)
                j_rand = rng.integers(n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                        
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                calls_made += 1
                
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    
                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
            
            # Check time inside loop to avoid long stalls
            if time.time() > deadline:
                break

        # Phase 2: Local Refinement
        # Use the remaining budget for local search around best_x
        # Since the function is step-like, a coordinate descent or small grid search is effective.
        
        # We will perform a coordinate descent with a shrinking step size.
        # Start with a relatively large step and shrink it.
        step_sizes = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
        
        current_x = best_x.copy()
        current_f = best_f
        
        for step in step_sizes:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
                
            improved = True
            while improved and calls_made < budget:
                improved = False
                for d in range(n):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                        
                    # Try positive direction
                    x_pos = current_x.copy()
                    x_pos[d] += step
                    x_pos = np.clip(x_pos, -5, 5)
                    f_pos = func(x_pos)
                    calls_made += 1
                    
                    if f_pos < current_f:
                        current_x = x_pos
                        current_f = f_pos
                        if current_f < best_f:
                            best_f = current_f
                            best_x = current_x.copy()
                        improved = True
                        continue # Restart loop from axis 0? Or continue? Usually restart or continue. Let's continue to next axis for efficiency, but restarting might be better for coordinate descent. Let's just continue.
                    
                    # Try negative direction
                    x_neg = current_x.copy()
                    x_neg[d] -= step
                    x_neg = np.clip(x_neg, -5, 5)
                    f_neg = func(x_neg)
                    calls_made += 1
                    
                    if f_neg < current_f:
                        current_x = x_neg
                        current_f = f_neg
                        if current_f < best_f:
                            best_f = current_f
                            best_x = current_x.copy()
                        improved = True
                        continue
                        
        return best_x, best_f
