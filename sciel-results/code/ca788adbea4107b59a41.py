"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        mu = lam // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0  # initial step size, covers good portion of box
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # not explicitly used, we use Cholesky
        C_1 = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute
        d_sigma = c_sigma * (1 + c_c + 2 * max(0, (c_c - 1) / c_c - 1))  # approximation

        best_x = x_mean.copy()
        best_f = float('inf')
        calls_used = 0

        for gen in range(budget):
            # Generate population
            offspring = []
            for i in range(lam):
                # Sample from N(0, C)
                z = np.random.randn(dim)
                y = x_mean + sigma * (C_1 @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate
            offspring_f = []
            for y in offspring:
                fy = func(y)
                calls_used += 1
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            offspring_f = np.array(offspring_f)

            # Sort by fitness
            idx = np.argsort(offspring_f)
            y_sorted = np.array([offspring[i] for i in idx])
            f_sorted = offspring_f[idx]

            # Update mean
            x_mean_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update covariance matrix
            # Compute rank-one and rank-mu updates
            y_mean = (x_mean_new - x_mean) / sigma

            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ y_mean

            # h_sigma check
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) / chi_n) < (1.4 + 2 / (dim + 1))

            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_mean

            # C update
            delta_h = (2 * h_sigma - 1) * c_c * (1 - c_c)
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C  # base decay
            # Actually standard formula:
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(y_mean, y_mean) + (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c) - delta_h * C)
            # Rank-mu update
            z_sorted = np.array([(y_sorted[i] - x_mean) / sigma for i in range(mu)])
            C += c_mu * np.sum(weights[:mu, None] * np.outer(z_sorted, z_sorted), axis=0)

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update C_1 (square root of C) via eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.maximum(eigvals, 1e-10)
            C_1 = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            x_mean = x_mean_new
            x_mean = np.clip(x_mean, -5, 5)

            # If budget exhausted, break
            if calls_used >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
