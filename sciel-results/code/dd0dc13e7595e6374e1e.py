# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid Differential Evolution + L-BFGS-B for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.perf_counter()
        deadline = t_start + 280.0
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        history = [] # Store (f, x) for top candidates

        def time_left():
            return deadline - time.perf_counter()

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
            # Keep top 10 in history for later polish
            history.append((f, x.copy()))
            history.sort(key=lambda t: t[0])
            if len(history) > 10:
                history.pop()

        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            update_best(x, f)
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution
        pop_size = 20
        F = 0.8 # Mutation strength
        CR = 0.9 # Crossover rate
        
        # Initialize population
        population = [rng.uniform(-5.0, 5.0, dim) for _ in range(pop_size)]
        fitness = [np.inf] * pop_size
        
        # Evaluate initial population
        for i in range(pop_size):
            if time_left() < 10.0 or calls_made >= budget:
                break
            fitness[i] = wrapped_func(population[i])
            
        # DE Loop
        while time_left() > 20.0 and calls_made < budget - 50: # Reserve 50 for polish
            # Sort population by fitness
            indices = np.argsort(fitness)
            population = [population[i] for i in indices]
            fitness = [fitness[i] for i in indices]
            
            # Generate mutants
            new_population = []
            new_fitness = []
            
            for i in range(pop_size):
                if time_left() < 10.0 or calls_made >= budget:
                    break
                    
                # Select 3 distinct individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                trial = np.clip(trial, -5.0, 5.0)
                
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(f_trial)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
                    
            population = new_population
            fitness = new_fitness

        # Phase 2: Local Polish with L-BFGS-B
        # Take the best points from history (which includes DE bests)
        # history is sorted by f, so history[0] is best.
        
        polished = set()
        
        for f_val, x_val in history[:5]: # Top 5 candidates
            if time_left() < 10.0 or calls_made >= budget:
                break
                
            x_key = tuple(x_val.round(6))
            if x_key in polished:
                continue
            polished.add(x_key)
            
            try:
                res = minimize(
                    wrapped_func,
                    x_val,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxls': 10, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # wrapped_func already updated best_x/best_f
            except Exception:
                pass

        # Phase 3: Random Restarts if budget remains
        while time_left() > 5.0 and calls_made < budget:
            x0 = rng.uniform(-5.0, 5.0, dim)
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 20, 'maxls': 5, 'ftol': 1e-10, 'gtol': 1e-6}
                )
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
