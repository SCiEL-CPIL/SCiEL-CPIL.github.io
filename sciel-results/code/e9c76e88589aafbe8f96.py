"""Optimizer using differential evolution with adaptive parameters for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer with adaptive strategy selection."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin from 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Population size: standard for DE is 5*n to 10*n, use 10*n for diversity
        pop_size = max(10, min(10 * n, budget // 2))
        # Ensure pop_size is odd for some strategies
        if pop_size % 2 == 0:
            pop_size += 1

        # Initialize population uniformly in [-5, 5]^n
        pop = rng.uniform(-5, 5, (pop_size, n))
        fitness = np.array([func(x) for x in pop])

        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # DE parameters
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate
        strategy = 'DE/rand/1/bin'

        # Adaptive parameters
        no_improve = 0
        eval_count = pop_size

        while eval_count < budget:
            # Check time
            if time.time() > deadline - 2:
                break

            # Sort population by fitness
            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]

            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = pop[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Adaptive strategy selection based on stagnation
            if no_improve > pop_size * 2:
                # Try a different strategy
                if strategy == 'DE/rand/1/bin':
                    strategy = 'DE/best/2/bin'
                    F = 0.7
                    CR = 0.8
                elif strategy == 'DE/best/2/bin':
                    strategy = 'DE/current-to-best/1/bin'
                    F = 0.6
                    CR = 0.7
                else:
                    strategy = 'DE/rand/1/bin'
                    F = 0.8
                    CR = 0.9
                no_improve = 0

            # Generate offspring for each individual
            for i in range(pop_size):
                if eval_count >= budget:
                    break
                if time.time() > deadline - 2:
                    break

                # Select strategy-specific parents
                if strategy == 'DE/rand/1/bin':
                    # DE/rand/1: v = x_rand1 + F * (x_rand2 - x_rand3)
                    while True:
                        r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                        if len({r1, r2, r3, i}) == 4:
                            break
                    v = pop[r1] + F * (pop[r2] - pop[r3])

                elif strategy == 'DE/best/2/bin':
                    # DE/best/2: v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
                    while True:
                        r1, r2, r3, r4 = rng.choice(pop_size, 4, replace=False)
                        if len({r1, r2, r3, r4, i}) == 5:
                            break
                    v = pop[0] + F * (pop[r1] - pop[r2]) + F * (pop[r3] - pop[r4])

                else:
                    # DE/current-to-best/1: v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
                    while True:
                        r1, r2 = rng.choice(pop_size, 2, replace=False)
                        if len({r1, r2, i}) == 3:
                            break
                    v = pop[i] + F * (pop[0] - pop[i]) + F * (pop[r1] - pop[r2])

                # Clip to bounds
                v = np.clip(v, -5, 5)

                # Binary crossover
                jrand = rng.integers(0, n)
                trial = pop[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]

                # Evaluate
                trial_f = func(trial)
                eval_count += 1

                # Selection
                if trial_f <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f

        return best_x, best_f
# EVOLVE-BLOCK-END
