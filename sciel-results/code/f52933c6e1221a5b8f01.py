"""Rewritten optimizer for BBOB f6 (Attractive sector) using a population-based CMA-ES variant."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based continuous optimizer using a simplified CMA-ES / evolution strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Initial population: mix of random points and a few structured points
        # Start with a small population to balance exploration and exploitation
        pop_size = 12  # number of offspring per generation
        n_gen = 0

        # Initial center: try a few promising starting points
        # For f6 (attractive sector), the optimum is often not at origin due to shift
        # Use a diverse initial set
        x_mean = np.zeros(dim)
        sigma = 1.5

        # Track best solution found
        x_best = None
        f_best = np.inf

        # Evaluate initial center and a few random points to get a better starting point
        candidates = []
        candidates.append(np.zeros(dim))
        candidates.append(rng.uniform(-4, 4, dim))
        candidates.append(rng.uniform(-3, 3, dim))
        candidates.append(rng.uniform(-2, 2, dim))
        candidates.append(rng.uniform(-5, 5, dim))

        for c in candidates:
            c = np.clip(c, -5, 5)
            fc = func(c)
            if fc < f_best:
                f_best = fc
                x_best = c.copy()
            n_gen += 1
            if n_gen >= budget:
                break

        if x_best is None:
            x_best = np.zeros(dim)
            f_best = func(x_best)
            n_gen += 1

        x_mean = x_best.copy()
        sigma = 1.0

        # Adaptive population size: start small, may increase
        # Use a simplified CMA-ES-like update
        # Covariance matrix approximation: use diagonal sigma^2 initially
        C = np.eye(dim)
        # Step size control
        sigma = 1.0

        calls_made = n_gen

        while calls_made < budget:
            # Generate offspring
            offspring = []
            offspring_f = []

            for i in range(pop_size):
                if calls_made >= budget:
                    break
                # Sample from multivariate normal with mean x_mean, cov sigma^2 * C
                # Simplified: use diagonal covariance (sigma^2 * I) for stability
                # But let's do a proper sample using Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)

                z = rng.standard_normal(dim)
                x_off = x_mean + sigma * L @ z
                x_off = np.clip(x_off, -5, 5)

                f_off = func(x_off)
                calls_made += 1

                offspring.append(x_off)
                offspring_f.append(f_off)

                if f_off < f_best:
                    f_best = f_off
                    x_best = x_off.copy()

            if len(offspring) == 0:
                break

            # Sort offspring by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Update mean: weighted average of best half
            lambda_ = max(1, pop_size // 2)
            mu = max(1, pop_size // 2)  # number of parents
            # Weights: linear
            weights = np.arange(1, mu + 1)
            weights = weights / weights.sum()

            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_offspring[i]
            x_mean = new_mean

            # Update covariance matrix (simplified CMA-ES update)
            # h_sigma: path length control
            # Simplified: just adapt sigma based on success rate
            success = sum(1 for i in range(mu) if sorted_f[i] < f_best) / mu
            if success > 0.5:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 10.0)

            # Update C: simple rank-mu update
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T) + cmu * sum(wi * (yi - ymean) @ (yi - ymean).T)
            # Simplified: use rank-1 update from best individual
            y = sorted_offspring[0] - x_mean
            c1 = 1 / (dim + 1)
            C = (1 - c1) * C + c1 * np.outer(y, y)

            # Ensure C is positive semi-definite and well-conditioned
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-8 * np.eye(dim)

            # Decay C back towards identity to prevent over-specialization
            C = 0.95 * C + 0.05 * np.eye(dim)

            n_gen += 1

        return x_best, f_best
# EVOLVE-BLOCK-END
