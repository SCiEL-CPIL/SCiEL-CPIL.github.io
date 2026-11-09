import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x = np.clip(x, -5, 5)
            v = func(x)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # 1. Initialization
        # Use LHS for diversity, covering the full domain
        pop_size = 10
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            points = sampler.random(n=pop_size)
            points = points * 10.0 - 5.0  # Scale to [-5, 5]
        except ImportError:
            points = rng.uniform(-5, 5, (pop_size, dim))

        population = []
        fitness = []
        
        for p in points:
            v = wrapped_func(p)
            population.append(p.copy())
            fitness.append(v)
            
        if best_x is None:
            best_x = population[0].copy()
            best_f = fitness[0]

        # 2. Differential Evolution (DE/rand/1)
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate
        
        while calls < self.budget - 5 and time.time() - start_time < time_limit - 10.0:
            improved = False
            for i in range(pop_size):
                if calls >= self.budget - 5 or time.time() - start_time > time_limit - 10.0:
                    break
                
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = population[i][j]
                
                # Ensure at least one dimension is mutated
                if np.all(trial == population[i]):
                    trial[rng.randint(dim)] = mutant[rng.randint(dim)]

                trial = np.clip(trial, -5, 5)
                
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    improved = True

            # Stagnation handling: if no improvement in a generation, perturb the worst
            if not improved:
                worst_idx = np.argmax(fitness)
                population[worst_idx] = population[worst_idx] + rng.normal(0, 0.5, dim)
                population[worst_idx] = np.clip(population[worst_idx], -5, 5)
                fitness[worst_idx] = wrapped_func(population[worst_idx])

        # 3. Final Polish with Nelder-Mead
        if best_x is not None and calls < self.budget - 10 and time.time() - start_time < time_limit - 15.0:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
