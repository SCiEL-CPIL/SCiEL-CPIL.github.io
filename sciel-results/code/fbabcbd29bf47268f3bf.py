import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Differential Evolution Parameters
        pop_size = 20
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            population.append((x, f))
            
        # Sort initial population
        population.sort(key=lambda p: p[1])
        
        while calls < budget and time.time() < deadline:
            # Generate new population
            new_population = []
            
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Select three distinct random individuals
                while True:
                    a, b, c = rng.integers(0, pop_size, 3)
                    if a != i and b != i and c != i and a != b and b != c and a != c:
                        break
                
                x_a = population[a][0]
                x_b = population[b][0]
                x_c = population[c][0]
                
                # Generate mutant vector
                v = x_a + F * (x_b - x_c)
                
                # Binomial crossover
                trial = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][0][j]
                
                trial = np.clip(trial, -5, 5)
                f_trial = eval_func(trial)
                
                # Selection
                if f_trial <= population[i][1]:
                    new_population.append((trial, f_trial))
                else:
                    new_population.append(population[i])
            
            if len(new_population) < pop_size:
                break
                
            population = new_population
            population.sort(key=lambda p: p[1])

        return best_x, best_f
