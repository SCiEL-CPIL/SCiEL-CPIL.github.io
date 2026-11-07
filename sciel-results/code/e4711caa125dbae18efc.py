import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        try:
            # Multi-island Differential Evolution with Nelder-Mead refinement
            num_islands = 3
            pop_size_per_island = 15
            F = 0.8  # Differential weight
            CR = 0.9  # Crossover rate
            migration_rate = 0.1  # Fraction of population to migrate
            nm_iterations = 10  # Nelder-Mead iterations per refinement
            
            # Initialize islands with diverse starting points
            islands = []
            for i in range(num_islands):
                center = rng.uniform(-4, 4, dim)
                population = []
                for j in range(pop_size_per_island):
                    x = center + rng.standard_normal(dim) * 1.5
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls_made += 1
                    population.append((x, fx))
                
                # Sort by fitness
                population.sort(key=lambda p: p[1])
                islands.append(population)
                
                # Update global best
                if population[0][1] < best_f:
                    best_f = population[0][1]
                    best_x = population[0][0].copy()
            
            # Main loop
            generation = 0
            while calls_made < budget and time.time() < deadline:
                generation += 1
                
                # For each island, perform DE
                for island_idx in range(num_islands):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    
                    pop = islands[island_idx]
                    new_pop = []
                    
                    # Keep the best individual
                    new_pop.append(pop[0])
                    
                    # Generate DE offspring for remaining slots
                    for i in range(1, pop_size_per_island):
                        if calls_made >= budget or time.time() >= deadline:
                            break
                        
                        # Select three distinct random individuals
                        r1, r2, r3 = rng.choice(len(pop), size=3, replace=False)
                        
                        # Create mutant vector
                        v = pop[r1][0] + F * (pop[r2][0] - pop[r3][0])
                        v = np.clip(v, -5, 5)
                        
                        # Crossover
                        trial = rng.random(dim) < CR
                        j_rand = rng.randint(dim)
                        trial[j_rand] = True
                        x_trial = np.where(trial, v, pop[i][0])
                        x_trial = np.clip(x_trial, -5, 5)
                        
                        # Evaluate
                        fx_trial = func(x_trial)
                        calls_made += 1
                        
                        # Selection
                        if fx_trial < pop[i][1]:
                            new_pop.append((x_trial, fx_trial))
                        else:
                            new_pop.append(pop[i])
                    
                    # Update island
                    new_pop.sort(key=lambda p: p[1])
                    islands[island_idx] = new_pop
                    
                    # Update global best
                    if new_pop[0][1] < best_f:
                        best_f = new_pop[0][1]
                        best_x = new_pop[0][0].copy()
                
                # Migration: exchange best individuals between islands
                if generation % 5 == 0 and calls_made < budget:
                    for i in range(num_islands):
                        if calls_made >= budget or time.time() >= deadline:
                            break
                        donor = (i + 1) % num_islands
                        if len(islands[donor]) > 0:
                            # Replace worst individual in island i with best from donor
                            if len(islands[i]) > 0:
                                islands[i][-1] = islands[donor][0].copy()
                
                # Periodic Nelder-Mead refinement on best individual
                if generation % 3 == 0 and calls_made < budget - 50:
                    if time.time() >= deadline:
                        break
                    
                    # Use Nelder-Mead on a copy of best_x
                    x0 = best_x.copy()
                    
                    # Wrap func to track calls and enforce deadline
                    def wrapped_func(x):
                        nonlocal calls_made
                        if calls_made >= budget or time.time() >= deadline:
                            return np.inf
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        calls_made += 1
                        return fx
                    
                    try:
                        result = minimize(wrapped_func, x0, method='Nelder-Mead', 
                                        options={'maxiter': nm_iterations, 'xatol': 1e-6, 'fatol': 1e-8})
                        if result.success and result.fun < best_f:
                            best_f = result.fun
                            best_x = np.clip(result.x, -5, 5)
                    except Exception:
                        pass
        
        except Exception:
            pass

        return best_x, best_f
