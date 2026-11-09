"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin

        rng = np.random.default_rng(42)

        # DE Parameters
        pop_size = max(20, 4 * dim)
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def _evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initialize population uniformly in [-5, 5]^dim
        population = []
        for i in range(pop_size):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f = _evaluate(x)
            population.append((x, f))

        if len(population) == 0:
            return best_x, best_f

        # Sort initial population
        population.sort(key=lambda x: x[1])

        # Main DE loop
        while call_count < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            new_population = list(population)
            
            for i in range(pop_size):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct individuals different from i
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != i and b != i and c != i and a != b and a != c:
                        break

                # Differential mutation
                v = population[a][0] + F * (population[b][0] - population[c][0])
                v = np.clip(v, -5.0, 5.0)

                # Binary crossover
                trial = population[i][0].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5.0, 5.0)
                f_trial = _evaluate(trial)

                # Selection
                if f_trial <= population[i][1]:
                    new_population[i] = (trial, f_trial)

            # Update population
            population = new_population
            population.sort(key=lambda x: x[1])

            # Adaptive F and CR based on progress
            if len(population) > 0:
                # Reduce F slightly as we converge
                F = max(0.5, F * 0.999)

        return best_x, best_f
# EVOLVE-BLOCK-END
