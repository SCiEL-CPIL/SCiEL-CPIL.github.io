import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        calls = 0

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with diverse sampling
        # Use a combination of structured and random sampling
        phase1_budget = min(150, budget // 3)
        
        # Halton sequence for quasi-random coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        for i in range(1, phase1_budget + 1):
            if calls >= phase1_budget or time.time() > deadline - 10:
                break
            point = np.array([halton(i, bases[j]) * 10.0 - 5.0 for j in range(n)])
            try_eval(point)

        # Add some random points for diversity
        random_budget = min(50, budget - calls)
        for _ in range(random_budget):
            if calls >= phase1_budget + random_budget or time.time() > deadline - 10:
                break
            try_eval(rng.uniform(-5, 5, n))

        # Phase 2: Local optimization using CMA-ES-like approach
        # Start from the best point found so far
        if best_x is None:
            return best_x, best_f

        # Use a population-based local search
        pop_size = min(15, budget - calls)
        if pop_size < 5:
            return best_x, best_f

        # Initialize population around best point
        sigma = 2.0
        population = []
        for _ in range(pop_size):
            x_new = best_x + rng.normal(0, sigma, n)
            f_new = try_eval(x_new)
            if f_new is not None:
                population.append((f_new, x_new.copy()))

        # Sort by fitness
        population.sort(key=lambda t: t[0])

        # Evolutionary loop
        max_generations = 20
        for gen in range(max_generations):
            if calls >= budget - 5 or time.time() > deadline - 5:
                break

            # Keep top half
            elite_count = max(2, pop_size // 2)
            population = population[:elite_count]

            # Update best
            if population[0][0] < best_f:
                best_f = population[0][0]
                best_x = population[0][1].copy()

            # Adapt sigma based on recent improvements
            if len(population) >= 2:
                improvement = population[0][0] - population[-1][0]
                if improvement > 0:
                    sigma *= 0.95
                else:
                    sigma *= 1.1
            sigma = np.clip(sigma, 0.1, 5.0)

            # Generate new offspring
            new_population = []
            for _ in range(pop_size):
                if calls >= budget - 5 or time.time() > deadline - 5:
                    break
                # Select parent with tournament
                idx1 = rng.integers(0, len(population))
                idx2 = rng.integers(0, len(population))
                if population[idx1][0] < population[idx2][0]:
                    parent = population[idx1][1]
                else:
                    parent = population[idx2][1]
                
                x_new = parent + rng.normal(0, sigma, n)
                f_new = try_eval(x_new)
                if f_new is not None:
                    new_population.append((f_new, x_new.copy()))

            # Merge and select
            population = population + new_population
            population.sort(key=lambda t: t[0])
            population = population[:pop_size]

        return best_x, best_f
