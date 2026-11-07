"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (mu+lambda)-ES with Sobol initialization and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Sobol-based initial sampling for good global coverage
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        n_init = 50
        init_points = sampler.random(n_init) * 10 - 5  # map to [-5, 5]

        best_x = init_points[0].copy()
        best_f = func(best_x)
        calls = 1

        for i in range(1, n_init):
            if time.time() > deadline:
                break
            x = init_points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # Population-based ES
        pop_size = 12
        mu = 6  # number of parents
        sigma = 1.5

        # Initialize population around best point
        rng = np.random.default_rng(42)
        population = [best_x.copy()]
        pop_f = [best_f]
        for i in range(1, pop_size):
            if time.time() > deadline:
                break
            x = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
            fx = func(x)
            calls += 1
            population.append(x)
            pop_f.append(fx)

        # Main ES loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Sort population by fitness
            sorted_idx = np.argsort(pop_f)
            population = [population[i] for i in sorted_idx]
            pop_f = [pop_f[i] for i in sorted_idx]

            # Take top mu as parents
            parents = population[:mu]
            parent_f = pop_f[:mu]

            # Compute weighted center of mass
            weights = np.arange(mu, 0, -1)
            weights /= weights.sum()
            center = sum(w * p for w, p in zip(weights, parents))

            # Compute sigma adaptation (1/5 rule on success rate)
            n_success = sum(1 for f in pop_f if f < parent_f[-1])  # improved over worst parent
            success_rate = n_success / len(pop_f)
            if success_rate > 0.2:
                sigma *= 1.1
            elif success_rate < 0.1:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 5.0)

            # Generate offspring
            new_population = []
            new_pop_f = []
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                parent = parents[i % mu]
                x = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls += 1
                new_population.append(x)
                new_pop_f.append(fx)

            # Combine with current population and select best pop_size
            all_x = population + new_population
            all_f = pop_f + new_pop_f

            sorted_idx = np.argsort(all_f)
            population = [all_x[i] for i in sorted_idx[:pop_size]]
            pop_f = [all_f[i] for i in sorted_idx[:pop_size]]

            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = population[0].copy()

        # Final polish: fine-grained local search around best point
        if calls < budget:
            sigma_polish = max(sigma * 0.1, 0.01)
            for _ in range(min(50, budget - calls)):
                if time.time() > deadline:
                    break
                x = np.clip(best_x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    sigma_polish *= 0.95
                else:
                    sigma_polish *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
