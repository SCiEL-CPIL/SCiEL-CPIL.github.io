"""Differential-Evolution style population optimizer for Rastrigin-rotated f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-style population optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = 20
        F = 0.8  # mutation scale
        CR = 0.5  # crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (lam, dim))
        fvals = np.array([func(x) for x in pop])

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        calls_made = lam

        # Main DE loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Find current best in population
            best_idx = np.argmin(fvals)
            if fvals[best_idx] < best_f:
                best_f = fvals[best_idx]
                best_x = pop[best_idx].copy()

            # For each individual, create mutant
            for i in range(lam):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select 3 distinct random indices different from i
                others = [j for j in range(lam) if j != i]
                a, b, c = rng.choice(others, size=3, replace=False)

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                # Evaluate
                ft = func(trial)
                calls_made += 1

                # Selection
                if ft <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = ft

            # Adapt F slightly: decrease over time for exploitation
            # But keep it high enough for exploration in multimodal landscape
            progress = calls_made / budget
            F = 0.8 * (1.0 - 0.3 * progress)  # 0.8 -> 0.56

        return best_x, best_f
# EVOLVE-BLOCK-END
