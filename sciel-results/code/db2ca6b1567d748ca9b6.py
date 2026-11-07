"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution (DE/rand/1/bin) optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # DE Parameters
        # Population size
        pop_size = max(10, 5 * n)
        # Scale factor F
        F = 0.8
        # Crossover rate CR
        CR = 0.9

        # Initialize population uniformly in [-5, 5]
        population = rng.uniform(-5, 5, (pop_size, n))
        
        # Evaluate initial population
        fitness = np.array([func(x) for x in population])
        calls = pop_size

        best_idx = np.argmin(fitness)
        best_f = fitness[best_idx]
        best_x = population[best_idx].copy()

        # Main loop
        while calls < budget:
            # Determine how many candidates we can evaluate in this generation
            # We evaluate one candidate per individual in the population.
            # If budget is exhausted, we stop.
            
            # We can only proceed if we have at least 1 call left.
            # In DE, we typically update the whole population.
            # To respect the budget strictly, we can do partial generations or just stop.
            # Since pop_size is small (25 for n=5), and budget is 500, we have ~20 generations.
            
            # If we don't have enough budget for a full generation, we can still evaluate
            # remaining individuals one by one until budget is exhausted.
            
            # For simplicity and correctness, let's evaluate the whole population each step,
            # but break if calls >= budget.
            
            # However, if calls + pop_size > budget, we might waste budget or exceed it.
            # The contract says "More than 500 calls ... invalid".
            # So we must ensure calls <= 500.
            
            # Let's just do full generations. If the last generation would exceed budget,
            # we stop before it.
            
            if calls + pop_size > budget:
                # We have some remaining budget. Let's use it for a single-step update?
                # Or just stop. Stopping is safer.
                # But we can use the remaining budget to evaluate a few more random points
                # or continue DE with fewer individuals?
                # Let's just stop. The best found so far is returned.
                break

            # Generate new population
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            for i in range(pop_size):
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Ensure they are distinct from i? Not strictly necessary for DE/rand/1/bin
                # but good practice.
                # rng.choice with replace=False ensures r1, r2, r3 are distinct.
                # They might equal i. That's fine.
                
                # Differential variation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = np.zeros_like(v)
                j_rand = rng.randint(n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

            # Update population
            population = new_population
            fitness = new_fitness

            # Update best
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_f:
                best_f = fitness[curr_best_idx]
                best_x = population[curr_best_idx].copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
