"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 10 for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        pop_size = 10
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (dim + 1.7) / ((dim + 9) * (dim * dim + 17))
        c_c = (4 + dim / dim) / (dim + 4 + 2 * dim / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * (mu_eff + 0.5) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        x_mean = rng.uniform(-4, 4, dim)
        sigma = 2.0
        C = np.eye(dim)
        L = np.eye(dim)  # Cholesky factor of C
        L_inv = np.eye(dim)  # Inverse of L
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        while calls < budget and time.time() < deadline:
            # Generate population using current C (via L)
            population = np.zeros((pop_size, dim))
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = L @ z
                population[i] = x_mean + sigma * y
                population[i] = np.clip(population[i], -5, 5)

            # Evaluate population
            fitness = np.full(pop_size, np.inf)
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                fitness[i] = func(population[i])
                calls += 1

            # Update best
            valid = fitness < np.inf
            if np.any(valid):
                min_idx = np.argmin(fitness)
                if fitness[min_idx] < best_f:
                    best_f = fitness[min_idx]
                    best_x = population[min_idx].copy()

            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            sorted_pop = population[sorted_indices]

            # Weighted mean of best mu individuals
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * sorted_pop[i]

            # Update step-size
            diff = x_new - x_mean
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-14:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ diff) / sigma
            else:
                p_sigma = (1 - c_sigma) * p_sigma

            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 100.0)

            # Update path for covariance
            if norm_diff > 1e-14:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (L_inv @ diff) / sigma

            # Compute h_sigma
            h_sigma = 1.0 if (1 - c_sigma) ** (1 / pop_size) > 0.999 else 0.0
            # More standard:
            h_sigma = (1 - c_sigma) ** (1 / pop_size)

            # Rank-one update
            C = (1 + c_1 * h_sigma * (2 - c_1) - c_1) * C + c_1 * np.outer(diff, diff)

            # Rank-mu update
            for i in range(mu):
                y_i = (sorted_pop[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2

            # Recompute Cholesky
            try:
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Add regularization
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                    L_inv = np.eye(dim)

            # Update mean
            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
