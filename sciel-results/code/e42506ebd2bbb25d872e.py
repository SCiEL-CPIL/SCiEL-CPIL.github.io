"""Multi-start Differential Evolution for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """DE/rand/1 with population of 50, multi-start diversity."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 50

        # Initialize population: mix of random points and some structured starts
        pop = np.zeros((pop_size, dim))
        # First few are structured to cover the space
        pop[0] = 0.0
        for i in range(1, pop_size):
            pop[i] = rng.uniform(-5, 5, dim)

        # Evaluate initial population
        fvals = np.array([func(p) for p in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        calls_used = pop_size

        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        # Track for adaptive strategies
        no_improve = 0

        while calls_used < budget:
            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Adaptive F and CR based on stagnation
            if no_improve > 20:
                F = 0.5
                CR = 0.5
            elif no_improve > 10:
                F = 0.7
                CR = 0.7
            else:
                F = 0.8
                CR = 0.9

            improved = False

            # Generate offspring for each member
            for i in range(pop_size):
                if calls_used >= budget:
                    break

                # Select three distinct random indices different from i
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != i and b != i and c != i and a != b and a != c:
                        break

                # Mutation: DE/rand/1
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                calls_used += 1

                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    improved = True

                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # If stuck for too long, inject diversity
            if no_improve > 30:
                # Replace worst half with new random points
                n_replace = pop_size // 2
                for i in range(pop_size - n_replace, pop_size):
                    if calls_used >= budget:
                        break
                    pop[i] = rng.uniform(-5, 5, dim)
                    fvals[i] = func(pop[i])
                    calls_used += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
