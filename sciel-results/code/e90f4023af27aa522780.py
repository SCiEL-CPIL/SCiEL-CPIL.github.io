"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based evolutionary strategy with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Population size
        pop_size = 20

        # Initial population: uniform in [-5, 5]
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitnesses = np.array([func(x) for x in population])

        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]

        # Adaptive step size
        sigma = 2.0
        success_count = 0
        trial_count = 0

        # Number of elites to keep
        elites = 5

        # Track calls made
        calls_made = pop_size

        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Select parents (top elites)
            sorted_indices = np.argsort(fitnesses)
            parents = population[sorted_indices[:elites]]

            # Generate offspring
            offspring = np.zeros((pop_size, dim))
            for i in range(pop_size):
                parent = parents[i % len(parents)]
                # Add noise
                noise = rng.standard_normal(dim)
                candidate = parent + sigma * noise
                # Clip to bounds
                candidate = np.clip(candidate, -5, 5)
                offspring[i] = candidate

            # Evaluate offspring
            offspring_fitness = np.array([func(x) for x in offspring])
            calls_made += pop_size

            # Combine parent and offspring populations
            combined = np.vstack([population, offspring])
            combined_fitness = np.concatenate([fitnesses, offspring_fitness])

            # Select next generation (top pop_size)
            sorted_indices = np.argsort(combined_fitness)
            next_population = combined[sorted_indices[:pop_size]]
            next_fitness = combined_fitness[sorted_indices[:pop_size]]

            # Update best
            global_best_idx = np.argmin(combined_fitness)
            if combined_fitness[global_best_idx] < best_f:
                best_x = combined[global_best_idx].copy()
                best_f = combined_fitness[global_best_idx]
                success_count += 1

            # Update step size based on success rate
            trial_count += 1
            if trial_count % 10 == 0:
                success_rate = success_count / (trial_count * pop_size)
                if success_rate > 0.15:
                    sigma *= 1.2
                elif success_rate < 0.05:
                    sigma *= 0.8
                else:
                    sigma *= 1.0
                success_count = 0
                trial_count = 0

            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)

            population = next_population
            fitnesses = next_fitness

        return best_x, best_f
# EVOLVE-BLOCK-END
