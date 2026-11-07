"""CMA-ES optimizer for BBOB-style Rastrigin function f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 30 for multimodal search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        # CMA-ES parameters
        lam = 30  # population size
        mu = lam // 2  # number of parents

        # Weights for the (mu, lambda)-CMA-ES
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Track best
        best_x = x_mean.copy()
        best_f = float('inf')

        # Generate initial population
        def sample_population():
            """Generate lam samples from N(x_mean, sigma^2 * C)."""
            population = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.standard_normal(dim)
                # Transform: B @ D @ z
                population[i] = x_mean + sigma * (B @ (D * z))
            return population

        # Evaluate initial population
        population = sample_population()
        fitness = np.zeros(lam)
        for i in range(lam):
            x_i = np.clip(population[i], -5, 5)
            fitness[i] = func(x_i)
            if fitness[i] < best_f:
                best_f = fitness[i]
                best_x = x_i.copy()

        calls_made = lam

        # Sort by fitness
        order = np.argsort(fitness)
        fitness_sorted = fitness[order]
        population_sorted = population[order]

        # Main loop
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Update mean
            x_mean_new = np.sum(weights[:, None] * population_sorted[:mu], axis=0)
            x_mean = x_mean_new

            # Update evolution path for covariance matrix
            pc = (1 - c_c) * pc + c_c * ((x_mean_new - x_mean) / (sigma * D))

            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / D * (x_mean_new - x_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) / chi_n) < (1.4 + 2 / (dim + 1))
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update covariance matrix
            delta_hs = (1 - hsig) * c_c * (2 - c_c)
            C = ((1 - c_1 - c_mu + delta_hs) * C + c_1 * (pc[:, None] * pc[None, :] + delta_hs * C)
                 + c_mu * np.sum(weights[:, None, None] * population_sorted[:mu] * population_sorted[:mu].T, axis=0)
                 - 2 * c_mu * np.sum(weights * population_sorted[:mu], axis=0)[:, None] * np.sum(weights * population_sorted[:mu], axis=0)[None, :])

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Generate next population
            population = sample_population()
            fitness = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                x_i = np.clip(population[i], -5, 5)
                fitness[i] = func(x_i)
                calls_made += 1
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = x_i.copy()

            # Sort
            order = np.argsort(fitness)
            fitness_sorted = fitness[order]
            population_sorted = population[order]

        return best_x, best_f
# EVOLVE-BLOCK-END
