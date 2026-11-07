"""Multi-start Differential Evolution optimizer for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallelized multi-start differential evolution for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Population size
        pop_size = 50
        # Differential evolution parameters
        F = 0.8       # mutation scale
        CR = 0.9      # crossover rate

        rng = np.random.default_rng(12345)

        # Initialize population with diverse points
        # Use stratified initialization for better coverage
        pop = np.zeros((pop_size, d))
        for i in range(pop_size):
            # Mix of random and structured initialization
            if i < d:
                # One point per dimension with focus
                pop[i] = rng.uniform(-5, 5, d)
                pop[i, i] = rng.uniform(-1, 1)
            elif i < 2 * d:
                # Points near origin
                pop[i] = rng.uniform(-2, 2, d)
            else:
                # Fully random
                pop[i] = rng.uniform(-5, 5, d)

        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # Track calls made
        calls_made = pop_size

        # Main DE loop
        while calls_made < B:
            # Check deadline
            if time.time() > deadline:
                break

            # Select best individual
            best_idx = np.argmin(fvals)
            best_x = pop[best_idx].copy()
            best_f = fvals[best_idx]

            # Generate offspring for each member
            for i in range(pop_size):
                if calls_made >= B:
                    break
                if time.time() > deadline:
                    break

                # DE/rand/1: select 3 distinct random individuals
                while True:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    if len(set([r1, r2, r3, i])) == 4:
                        break

                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                # Clip to bounds
                v = np.clip(v, -5, 5)

                # Crossover
                trial = np.empty(d)
                j_rand = rng.integers(0, d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Evaluate trial
                ftrial = func(trial)
                calls_made += 1

                # Selection
                if ftrial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = ftrial
                    if ftrial < best_f:
                        best_f = ftrial
                        best_x = trial.copy()

            # Periodic re-injection to maintain diversity
            if calls_made < B and pop_size > 1:
                # Replace worst individual with a new random point every ~50 evaluations
                if calls_made % 50 == 0:
                    worst_idx = np.argmax(fvals)
                    new_point = rng.uniform(-5, 5, d)
                    fnew = func(new_point)
                    calls_made += 1
                    pop[worst_idx] = new_point
                    fvals[worst_idx] = fnew
                    if fnew < best_f:
                        best_f = fnew
                        best_x = new_point.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
