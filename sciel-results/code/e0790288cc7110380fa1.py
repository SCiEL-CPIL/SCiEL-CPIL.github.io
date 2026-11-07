"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Cross-Entropy Method with adaptive population for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CEM parameters
        pop_size = 50
        elite_ratio = 0.2  # top 20% of population
        elite_size = max(5, int(pop_size * elite_ratio))

        # Initialize distribution parameters
        mean = np.zeros(dim)
        std = np.ones(dim) * 3.0  # start with wide spread

        best_x = None
        best_f = float('inf')
        budget_used = 0

        # Track convergence
        no_improve_count = 0
        convergence_threshold = 1e-6

        while budget_used < budget:
            if time.time() > deadline:
                break

            # Sample population from current distribution
            samples = mean + std * rng.standard_normal((pop_size, dim))
            samples = np.clip(samples, -5, 5)

            # Evaluate all samples
            fitness = np.array([func(samples[i]) for i in range(pop_size)])
            budget_used += pop_size

            # Update best
            idx_best = np.argmin(fitness)
            if fitness[idx_best] < best_f:
                best_f = fitness[idx_best]
                best_x = samples[idx_best].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Check for convergence
            if np.std(fitness) < convergence_threshold:
                # Reset distribution to explore more
                mean = best_x.copy()
                std = np.ones(dim) * 0.5
                no_improve_count = 0
                continue

            # Select elite individuals
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_samples = samples[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Update distribution parameters
            new_mean = np.mean(elite_samples, axis=0)
            new_std = np.std(elite_samples, axis=0)

            # Smooth update to avoid abrupt changes
            alpha = 0.7
            mean = alpha * new_mean + (1 - alpha) * mean
            std = alpha * np.maximum(new_std, 0.01) + (1 - alpha) * std

            # Ensure std doesn't become too small or too large
            std = np.clip(std, 0.01, 5.0)

            # If stagnating for too long, try a different strategy
            if no_improve_count > 10:
                # Reduce std to focus search
                std *= 0.5
                std = np.maximum(std, 0.01)
                no_improve_count = 0

        # Final refinement with (1+1)-ES on the best point
        if best_x is not None and budget_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1

            while budget_used < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_f:
                        best_x, best_f = x, fx
                else:
                    sigma *= 0.95

                sigma = max(0.001, min(sigma, 3.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
