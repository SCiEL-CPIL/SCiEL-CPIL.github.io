"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution (DE/rand/1/bin) for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size for DE
        pop_size = 20
        
        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        pop_f = np.array([func(x) for x in pop])
        calls_made = pop_size

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_fx = pop_f[best_idx]

        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        # Adaptive F and CR based on generation
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate trial vector for each member
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Mutation: DE/rand/1
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover: binomial
                r = rng.random()
                j_rand = rng.integers(0, dim)
                trial = np.where(rng.random(dim) < CR, v, pop[i])
                trial[j_rand] = v[j_rand]  # Ensure at least one gene from v

                # Evaluate
                trial_f = func(trial)
                calls_made += 1

                # Selection
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_fx:
                        best_fx = trial_f
                        best_x = trial.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
