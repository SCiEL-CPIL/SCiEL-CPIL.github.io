"""Optimizer for BBOB-style f15 (Rastrigin rotated) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start from center of the box
        x = np.zeros(dim)
        f = func(x)
        best_x = x.copy()
        best_f = f
        call_count = 1

        # CMA-ES parameters
        sigma = 2.0
        # Initial covariance matrix: identity
        C = np.eye(dim)
        # Inverse covariance
        C_inv = np.eye(dim)
        # Square root of C
        C_sqrt = np.eye(dim)

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))

        # Step size control (1/5 rule variant for CMA)
        # Standard CMA-ES parameters
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        eigvals = np.maximum(eigvals, 1e-12)

        while call_count + lambda_ <= budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Update square root of C
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Generate population
            z = np.random.randn(lambda_, dim)
            pop = x + sigma * (z @ C_sqrt.T)  # lambda_ x dim
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            pop_f = np.array([func(pop[i]) for i in range(lambda_)])
            call_count += lambda_

            # Update best
            for i in range(lambda_):
                if pop_f[i] < best_f:
                    best_f = pop_f[i]
                    best_x = pop[i].copy()

            # Sort
            idx = np.argsort(pop_f)
            y = pop[idx]
            y_f = pop_f[idx]

            # Weighted mean
            y_mean = (weights[:, np.newaxis] * y[:mu]).sum(axis=0)

            # Update evolution paths
            d_norm = np.sqrt(dim * (2 - np.exp(-2 / (dim + 2))))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt.T @ (y_mean - x) / sigma
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count - lambda_))) <
                     (1 + 2 / (dim + 1)) * d_norm)

            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mean - x) / sigma

            # Update covariance
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z_i = C_sqrt.T @ (y[i] - x) / sigma
                rank_mu += weights[i] * np.outer(z_i, z_i)

            C = (1 + c_1 * (c_1 + c_mu - 1) + delta_h) * C - c_1 * rank_one + c_mu * rank_mu

            # Update mean
            x = y_mean

            # Update sigma
            d_sigma = np.sqrt((1 - (1 - c_sigma) ** (2 * (call_count - lambda_))) * (1 - (1 - c_c) ** (2 * (call_count - lambda_))) * (1 + 2 / (dim + 1)))
            sigma *= np.exp(c_sigma / d_sigma) * (np.linalg.norm(C_sqrt.T @ p_sigma) / d_norm - 1)

            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)

        return best_x, best_f
# EVOLVE-BLOCK-END
