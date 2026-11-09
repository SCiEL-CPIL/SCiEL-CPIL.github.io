"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution optimizer with error-repair for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # DE Parameters
        pop_size = 10
        F = 0.8       # Differential weight
        CR = 0.9      # Crossover rate
        bounds_low = -5.0
        bounds_high = 5.0

        # Initialize population
        # Spread initial population to cover the box
        population = np.random.uniform(bounds_low, bounds_high, (pop_size, dim))
        fitness = np.array([func(x) for x in population])
        calls = pop_size

        # Track global best
        best_idx = np.argmin(fitness)
        global_best_x = population[best_idx].copy()
        global_best_f = fitness[best_idx]

        def check_time():
            return (time.time() - start_time) < time_limit

        def evaluate(x):
            nonlocal calls, global_best_x, global_best_f
            x = np.clip(x, bounds_low, bounds_high)
            f = func(x)
            calls += 1
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()
            return f

        # Main DE loop
        while check_time() and calls + pop_size <= budget:
            # Generate offspring
            new_population = np.empty_like(population)
            new_fitness = np.empty(pop_size)
            
            for i in range(pop_size):
                if not check_time() or calls >= budget:
                    break
                    
                # DE/rand/1 mutation
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                # Ensure r1, r2 != i
                while r1 == i:
                    r1 = rng.choice(pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.choice(pop_size)
                    
                v = population[r1] + F * (population[r2] - population[i])
                
                # Binomial crossover
                j_rand = rng.integers(dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Error Repair: If trial is out of bounds, reflect it back
                # This is the 'error_repair' mechanism to test
                for j in range(dim):
                    if trial[j] < bounds_low:
                        trial[j] = bounds_low + (bounds_low - trial[j])
                    elif trial[j] > bounds_high:
                        trial[j] = bounds_high - (trial[j] - bounds_high)
                
                # Clip just in case (double safety)
                trial = np.clip(trial, bounds_low, bounds_high)
                
                f_trial = evaluate(trial)
                
                # Selection: Greedy tournament (1-vs-1)
                if f_trial <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

            # Update population
            population = new_population
            fitness = new_fitness

            # Update global best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < global_best_f:
                global_best_f = fitness[best_idx]
                global_best_x = population[best_idx].copy()

        # Use remaining budget for local refinement around the best point
        # Simple local search to squeeze out more accuracy
        if check_time() and calls < budget:
            x = global_best_x.copy()
            f = global_best_f
            sigma = 0.5
            for _ in range(10):
                if not check_time() or calls >= budget:
                    break
                for _ in range(5):
                    if not check_time() or calls >= budget:
                        break
                    z = rng.normal(0, sigma, dim)
                    x_trial = np.clip(x + z, bounds_low, bounds_high)
                    f_trial = evaluate(x_trial)
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        sigma *= 1.2  # Expand slightly on success
                    else:
                        sigma *= 0.8  # Shrink on failure
                        if sigma < 0.01:
                            break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
