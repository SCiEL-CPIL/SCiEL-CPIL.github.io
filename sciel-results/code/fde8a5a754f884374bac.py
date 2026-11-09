"""DE based black-box optimizer for the rotated Sphere (BBOB f1)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution for 5-D continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # DE parameters
        # Population size: typically 5*dim to 10*dim, but budget is tight.
        # Let's try NP=10 for d=5.
        NP = 10
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        # Uniform in [-5, 5]
        pop = rng.uniform(-5.0, 5.0, (NP, d))
        
        # Evaluate initial population
        fitness = np.empty(NP)
        calls = 0
        for i in range(NP):
            fitness[i] = func(pop[i])
            calls += 1
            if calls >= budget:
                break
        
        # Track best
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        if calls >= budget:
            return best_pos, best_fit

        # Main loop
        while calls < budget:
            # Check if we can afford a full generation
            if budget - calls < NP:
                # If we can't do a full generation, use remaining budget for random local search around best
                # Or just break and let the evaluator handle the "repeat last n_i"
                # It's better to use the remaining calls for something useful.
                # Let's do a simple local search: perturb best_pos slightly.
                while calls < budget:
                    # Small random perturbation
                    step = 0.1 * rng.standard_normal(d)
                    new_pos = np.clip(best_pos + step, -5.0, 5.0)
                    f_new = func(new_pos)
                    calls += 1
                    if f_new < best_fit:
                        best_fit = f_new
                        best_pos = new_pos
                break

            # Standard DE step
            for i in range(NP):
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                while r1 == r2 or r1 == r3 or r2 == r3:
                    r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                
                # Create trial vector
                # v = pop[r1] + F * (pop[r2] - pop[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                # For each dimension, decide whether to take from v or pop[i]
                # Standard DE/rand/1/bin
                trial = np.empty(d)
                j_rand = rng.integers(0, d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to box
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate
                calls += 1
                if calls >= budget:
                    f_trial = func(trial)
                    if f_trial < fitness[i]:
                        fitness[i] = f_trial
                        pop[i] = trial
                    if f_trial < best_fit:
                        best_fit = f_trial
                        best_pos = trial.copy()
                    break
                
                f_trial = func(trial)
                
                # Selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < best_fit:
                        best_fit = f_trial
                        best_pos = trial.copy()
            
            if calls >= budget:
                break

        return best_pos, best_fit
# EVOLVE-BLOCK-END
