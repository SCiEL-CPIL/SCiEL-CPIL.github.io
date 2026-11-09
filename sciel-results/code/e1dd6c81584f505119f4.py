"""Rewritten program: CMA-ES with population-based search for BBOB f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size lambda=12 for global exploration of Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initial population: spread across the box
        lam = 12
        # Initial sigma large to cover the space
        sigma = 2.5

        # Initial mean: center of the box
        mean = np.zeros(dim)

        # Initial covariance: identity
        C = np.eye(dim)
        # Precompute sqrt of C (Cholesky) for sampling
        Cmat = np.linalg.cholesky(C)

        # Evaluate initial population
        population = []
        for i in range(lam):
            z = rng.standard_normal(dim)
            x = np.clip(mean + sigma * (Cmat @ z), -5, 5)
            fx = func(x)
            population.append((x, fx))

        best_x, best_f = min(population, key=lambda t: t[1])
        calls = lam

        # CMA-ES parameters
        mu = lam // 2  # number of parents
        # Weights: standard CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma

        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Evolution path for sigma
        ps = np.zeros(dim)
        # Evolution path for C
        pc = np.zeros(dim)

        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls < budget:
            if time.time() > deadline:
                break

            # Sort population by fitness
            population.sort(key=lambda t: t[1])

            # Update best
            if population[0][1] < best_f:
                best_x, best_f = population[0][0], population[0][1]

            # Select parents
            parents = [p[0] for p in population[:mu]]
            parents = np.array(parents)

            # Weighted mean of parents
            new_mean = weights @ parents

            # Update evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / \
                 np.linalg.norm(Cmat) * (Cmat.T @ ((new_mean - mean) / sigma))

            # Check for step size increase
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) < \
               (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update evolution path for C
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                 (Cmat.T @ ((new_mean - mean) / sigma))

            # Rank-one and rank-mu update for C
            delta_sigma = (1 - c_c) * (np.linalg.norm(pc) ** 2 / dim - 1)
            C = (1 + c_1 * delta_sigma + c_mu - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc) \
                + c_mu / sigma ** 2 * sum(
                    w_i * np.outer(
                        (parents[i] - mean) / sigma,
                        (parents[i] - mean) / sigma
                    ) for i, w_i in enumerate(weights)
                )

            # Regularize C to ensure positive definiteness
            # Add small identity if needed
            C = 0.5 * (C + C.T)  # symmetrize
            # Check eigenvalues
            try:
                Cmat = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Fallback: reset C to identity if it becomes non-PD
                C = np.eye(dim)
                Cmat = np.eye(dim)
                pc = np.zeros(dim)

            mean = new_mean

            # Generate new population
            population = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x = np.clip(mean + sigma * (Cmat @ z), -5, 5)
                fx = func(x)
                population.append((x, fx))
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

        # Final best
        if population:
            population.sort(key=lambda t: t[1])
            if population[0][1] < best_f:
                best_x, best_f = population[0][0], population[0][1]

        return best_x, best_f
# EVOLVE-BLOCK-END
