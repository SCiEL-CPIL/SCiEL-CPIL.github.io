"""Bootstrap spawn: Differential Evolution for 5D Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for 5D Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return float(func(x))

        def time_ok():
            return calls[0] < budget and time.time() < deadline

        # Parameters
        pop_size = 12  # Small population to allow more generations
        F = 0.8        # Scale factor
        CR = 0.5       # Crossover rate

        # Initialize population uniformly in [-5, 5]
        # We also add the origin as a candidate since the shift is in [-4, 4]
        pop = rng.uniform(-5, 5, (pop_size, dim))
        # Ensure at least one point is near the center? 
        # Actually, uniform is fine for Rastrigin as it's periodic.
        
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if not time_ok():
                break
            fitness[i] = call_func(pop[i])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # Main DE Loop
        while time_ok():
            for i in range(pop_size):
                if not time_ok():
                    break
                
                # Select 3 distinct random indices different from i
                # In a small population, we need to be careful
                # If pop_size is small, we might pick duplicates if not careful, 
                # but DE/rand/1 requires distinct r1, r2, r3 != i.
                # With pop_size=12, this is easy.
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                # Ensure they are not i
                if r1 == i: r1 = (r1 + 1) % pop_size
                if r2 == i: r2 = (r2 + 1) % pop_size
                if r3 == i: r3 = (r3 + 1) % pop_size
                # Ensure distinctness after adjustment is not strictly required for DE performance 
                # but let's try to keep them distinct from each other too if possible.
                # For simplicity and speed, we just proceed.

                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluation
                f_trial = call_func(trial)
                
                # Selection
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        # Final Polish: Coordinate Descent with adaptive step
        # This can quickly refine the solution if we are in the global basin.
        x = best_x.copy()
        fx = best_f
        step = 0.1
        for _ in range(20): # Max 20 iterations of polish
            if not time_ok():
                break
            improved = False
            for d in range(dim):
                for sign in [1, -1]:
                    x_trial = x.copy()
                    x_trial[d] += sign * step
                    x_trial = np.clip(x_trial, -5, 5)
                    fx_trial = call_func(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                        break # Move in the best direction found for this dim
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
            else:
                # Keep step same or slightly increase? No, shrink to converge
                step *= 0.8 

        return x, fx
# EVOLVE-BLOCK-END
