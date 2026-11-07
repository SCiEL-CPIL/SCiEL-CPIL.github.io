"""Multi-fidelity population-based ES with adaptive step size for BBOB f9."""
import numpy as np


class Optimizer:
    """Population-based evolutionary strategy with adaptive sigma and archive."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize population
        pop_size = 20
        pop = np.array([rng.uniform(-5, 5, dim) for _ in range(pop_size)])
        fitness = np.array([func(x) for x in pop])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # Adaptive sigma: start large for global exploration
        sigma = 3.0
        sigma_min = 1e-8
        sigma_max = 10.0

        # Track improvement rate for adaptive scaling
        prev_best_f = best_f
        improvement_rate = 1.0  # initial rate

        calls_made = pop_size
        remaining = budget - calls_made

        # Phase 1: Global exploration (first 40% of budget)
        # Phase 2: Transition (next 30%)
        # Phase 3: Fine exploitation (last 30%)

        while remaining > 0:
            # Determine current phase
            progress = 1.0 - remaining / max(budget, 1)
            if progress < 0.4:
                phase = 1
            elif progress < 0.7:
                phase = 2
            else:
                phase = 3

            # Adapt sigma based on phase and improvement rate
            if phase == 1:
                # Large steps for global exploration
                target_sigma = 3.0
            elif phase == 2:
                # Medium steps
                target_sigma = 1.0
            else:
                # Small steps for exploitation
                target_sigma = 0.3

            # Scale by improvement rate: if improving fast, keep sigma; if slow, reduce
            if improvement_rate < 0.1:
                sigma = max(sigma * 0.7, sigma_min)
            elif improvement_rate > 0.5:
                sigma = min(sigma * 1.2, sigma_max)
            else:
                # Move toward target sigma
                sigma = sigma * 0.8 + target_sigma * 0.2

            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Generate offspring
            n_offspring = min(pop_size, remaining)
            remaining -= n_offspring

            # Select parents: use top half of population for diversity
            sorted_idx = np.argsort(fitness)
            parent_indices = sorted_idx[:max(1, pop_size // 2)]

            offspring = np.zeros((n_offspring, dim))
            offspring_fitness = np.zeros(n_offspring)

            for i in range(n_offspring):
                # Pick a random parent from the elite
                pi = parent_indices[rng.integers(len(parent_indices))]
                # Add Gaussian perturbation
                mutant = pop[pi] + sigma * rng.standard_normal(dim)
                mutant = np.clip(mutant, -5, 5)
                offspring[i] = mutant
                offspring_fitness[i] = func(mutant)

            # Update population: keep best individuals
            # Combine population and offspring
            all_x = np.vstack([pop, offspring])
            all_f = np.concatenate([fitness, offspring_fitness])

            # Keep top pop_size
            keep_idx = np.argsort(all_f)[:pop_size]
            pop = all_x[keep_idx].copy()
            fitness = all_f[keep_idx].copy()

            # Update best
            new_best_idx = np.argmin(fitness)
            new_best_f = fitness[new_best_idx]

            if new_best_f < best_f:
                improvement = best_f - new_best_f
                # Normalize improvement rate
                if best_f > 1e-6:
                    improvement_rate = min(1.0, improvement / max(abs(best_f), 1e-8))
                else:
                    improvement_rate = 1.0 if improvement > 0 else 0.0
                best_f = new_best_f
                best_x = pop[new_best_idx].copy()

            # If no improvement, decay improvement rate
            else:
                improvement_rate *= 0.9

            # In exploitation phase, also try local refinement around best
            if phase == 3 and remaining > 0:
                local_sigma = sigma * 0.3
                local_mutant = best_x + local_sigma * rng.standard_normal(dim)
                local_mutant = np.clip(local_mutant, -5, 5)
                local_f = func(local_mutant)
                remaining -= 1

                if local_f < best_f:
                    best_f = local_f
                    best_x = local_mutant.copy()
                    improvement_rate = 0.5

        return best_x, best_f
