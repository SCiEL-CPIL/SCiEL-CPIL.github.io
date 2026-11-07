"""Differential Evolution optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Pure DE with adaptive mutation factor for global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Large static population for global exploration
        pop_size = 50
        CR = 0.5  # Crossover rate
        F = 0.8   # Fixed mutation factor

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # Track calls used
        calls_made = pop_size

        # Main DE loop
        while calls_made < budget:
            for i in range(pop_size):
                if calls_made >= budget:
                    break

                # Select three distinct random individuals different from i
                # Using choice with replace=False on a filtered list is slightly slow but dim is small.
                # For efficiency, we can just pick 3 unique indices and ensure they are not i.
                # Given pop_size=50, picking 3 distinct is easy.
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                # Ensure none are i. If any is i, we can re-pick or just use different ones.
                # Simplest robust way for small N:
                # But rng.choice(pop_size, 3, replace=False) might include i.
                # Let's use a helper or just check.
                # Actually, for DE, using i is not allowed in the standard definition (distinct).
                # Let's just re-sample if collision happens, or use the list comprehension method which is safe.
                # Given performance constraints are loose (300s), the list comprehension is fine.
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])

                # Crossover: binomial crossover
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                ft = func(trial)
                calls_made += 1

                # Selection: keep the better
                if ft <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
