import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315  # 15s safety margin from 330s limit
        calls = 0
        max_calls = self.budget
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            calls += 1
            # Ensure x is a numpy array
            x = np.asarray(x, dtype=float)
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Differential Evolution Parameters
        # Population size typically 5*dim to 10*dim
        pop_size = 25
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        # Uniformly distribute in [-5, 5]^dim
        population = rng.uniform(-5, 5, size=(pop_size, self.dim))
        
        # Evaluate initial population
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            try:
                pop_fitness[i] = wrapper(population[i])
            except (ValueError, TimeoutError):
                break

        # If we haven't filled the population due to limits, stop
        if np.any(np.isinf(pop_fitness)):
            return best_x, best_f

        # DE Main Loop
        # We run until budget or time is exhausted
        while True:
            # Check global constraints before generating new candidates
            if calls >= max_calls or time.time() > deadline:
                break

            # For each member of the population
            for i in range(pop_size):
                # Check constraints again
                if calls >= max_calls or time.time() > deadline:
                    break

                # Select two distinct indices different from i
                # We need 3 distinct indices for DE/rand/1
                # Since pop_size >= 5, this is safe
                j, k = rng.choice(pop_size, size=2, replace=False)
                while j == i:
                    j = rng.choice(pop_size)
                while k == i or k == j:
                    k = rng.choice(pop_size)

                # Mutation: v = x_j + F * (x_k - x_i)
                v = population[j] + F * (population[k] - population[i])
                
                # Clip to box [-5, 5]
                v = np.clip(v, -5, 5)

                # Crossover: u = v with probability CR, else x_i
                # Binary mask
                mask = rng.random(self.dim) < CR
                # Ensure at least one gene is from mutant v (standard DE practice)
                if not np.any(mask):
                    idx = rng.integers(self.dim)
                    mask[idx] = True
                    
                u = np.where(mask, v, population[i])

                # Evaluate trial vector
                try:
                    f_u = wrapper(u)
                except (ValueError, TimeoutError):
                    break

                # Selection
                if f_u <= pop_fitness[i]:
                    population[i] = u
                    pop_fitness[i] = f_u

        if best_x is None:
            # Fallback
            best_x = np.zeros(self.dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf

        return best_x, best_f
