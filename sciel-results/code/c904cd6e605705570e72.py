"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based Differential Evolution (DE) for Rosenbrock rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size tuned for budget=500, dim=5
        # Use a moderate population to balance exploration and exploitation
        pop_size = 25
        n_evals = 0
        max_evals = budget

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.empty(pop_size)
        for i in range(pop_size):
            fvals[i] = func(pop[i])
            n_evals += 1
            if n_evals >= max_evals:
                break

        # Adaptive DE parameters
        F = 0.7  # Scale factor - slightly lower for more exploitation
        CR = 0.8  # Crossover rate - slightly lower to maintain more of parent structure
        best_x = pop[np.argmin(fvals)].copy()
        best_f = fvals[np.argmin(fvals)]

        # Track best for early termination check
        while n_evals < max_evals:
            # Get best individual
            best_idx = np.argmin(fvals)
            best_x = pop[best_idx].copy()
            best_f = fvals[best_idx]

            # Generate offspring for each member
            for i in range(pop_size):
                if n_evals >= max_evals:
                    break

                # Select three distinct random indices different from i
                mask = np.ones(pop_size, dtype=bool)
                mask[i] = False
                candidates = np.where(mask)[0]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # Use DE/rand/1 with best bias
                # v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])

                # Blend with best to guide towards known good region
                # This helps navigate the Rosenbrock valley
                v = 0.7 * v + 0.3 * best_x

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                n_evals += 1

                # Tournament-style selection: keep better
                if f_trial < fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial

                # Update global best
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

            # Adaptive parameter adjustment: reduce F as we progress
            # to increase exploitation in later stages
            progress = n_evals / max_evals
            F = 0.9 - 0.4 * progress  # F decreases from 0.9 to 0.5
            CR = 0.7 + 0.3 * (1 - progress)  # CR decreases from 1.0 to 0.7

        # Return the best found
        best_idx = np.argmin(fvals)
        return pop[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END
