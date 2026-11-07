import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        try:
            # Differential Evolution parameters
            pop_size = 20
            F = 0.8  # Scale factor
            CR = 0.5  # Crossover rate
            max_generations = 25
            
            # Initialize population uniformly in [-5, 5]^dim
            population = rng.uniform(-5, 5, (pop_size, dim))
            fitness = np.full(pop_size, np.inf)
            
            # Evaluate initial population
            for i in range(pop_size):
                if calls_made >= budget or time.time() >= deadline:
                    break
                fitness[i] = func(population[i])
                calls_made += 1
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = population[i].copy()
            
            if calls_made >= budget or time.time() >= deadline:
                return best_x, best_f
            
            # Main DE loop
            for gen in range(max_generations):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Sort population by fitness
                order = np.argsort(fitness)
                population = population[order]
                fitness = fitness[order]
                
                # Update best
                if fitness[0] < best_f:
                    best_f = fitness[0]
                    best_x = population[0].copy()
                
                new_population = np.copy(population)
                new_fitness = np.copy(fitness)
                
                for i in range(pop_size):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    
                    # Select three distinct random indices different from i
                    while True:
                        a, b, c = rng.choice(pop_size, size=3, replace=False)
                        if len(set([a, b, c, i])) == 4:
                            break
                    
                    # DE/rand/1 mutation
                    mutant = population[a] + F * (population[b] - population[c])
                    mutant = np.clip(mutant, -5, 5)
                    
                    # Binomial crossover
                    trial = np.copy(population[i])
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    
                    # Evaluate trial
                    f_trial = func(trial)
                    calls_made += 1
                    
                    # Selection
                    if f_trial <= new_fitness[i]:
                        new_population[i] = trial
                        new_fitness[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                
                population = new_population
                fitness = new_fitness
                
                # Check for convergence
                if np.std(fitness[:5]) < 1e-10:
                    break

        except Exception:
            pass

        return best_x, best_f
