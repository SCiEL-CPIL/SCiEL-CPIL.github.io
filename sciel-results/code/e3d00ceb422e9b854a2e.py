import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2

        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Initial parameters
        x0 = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Adaptation parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        hsig = (1.4 + 2 / (dim + 1)) * (1 - np.exp(-2 * (1 - cc) / dim))

        # Cholesky decomposition of C
        B = np.linalg.cholesky(C)
        D = np.sqrt(np.diag(B))

        # Track best
        best_x = x0.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        while calls < budget:
            # Generate population
            population = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x0 + sigma * (B @ (D * z))
                population[i] = np.clip(y, -5, 5)

            # Evaluate population
            fitness = np.array([func(population[i]) for i in range(lam)])
            calls += lam

            if calls >= budget:
                # Check if any in this population beat best
                for i in range(lam):
                    if fitness[i] < best_f:
                        best_f = fitness[i]
                        best_x = population[i].copy()
                break

            # Sort by fitness
            idx = np.argsort(fitness)
            sorted_pop = population[idx]
            sorted_fit = fitness[idx]

            # Update best
            if sorted_fit[0] < best_f:
                best_f = sorted_fit[0]
                best_x = sorted_pop[0].copy()

            # Recombination: weighted mean of top mu
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (sorted_pop[i] - x0)

            # Update evolution paths
            if sigma > 1e-20:
                pc = (1 - cc) * pc + cc * np.sqrt(mueff) * (B @ (D * (y_w / sigma)))
                ps = (1 - cs) * ps + cs * np.sqrt(mueff) * (B @ (D * (y_w / sigma)))
            else:
                pc = (1 - cc) * pc
                ps = (1 - cs) * ps

            # Update sigma
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(cs / dsigma * (norm_ps / np.sqrt(1 - (1 - cs) ** 2) - 1))

            # Update covariance matrix
            norm_pc = np.linalg.norm(pc)
            if norm_pc / np.sqrt(1 - (1 - cc) ** 2) < (1.4 + 2 / (dim + 1)) * hsig:
                delta_hs = 1.0
                rank_one = np.outer(pc, pc)
            else:
                delta_hs = 0.0
                rank_one = np.outer(pc, pc) + np.eye(dim)

            C = (1 + c1 * delta_hs - c1 - cmu) * C + c1 * rank_one
            for i in range(mu):
                if sigma > 1e-20:
                    yi = (sorted_pop[i] - x0) / sigma
                else:
                    yi = np.zeros(dim)
                C += cmu * weights[i] * np.outer(yi, yi)

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            x0 = x0 + sigma * y_w

            # Update Cholesky
            B = np.linalg.cholesky(C)
            D = np.sqrt(np.diag(B))

        return best_x, best_f
