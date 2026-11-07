"""Optimized continuous optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-to-local strategy: uniform sampling + CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Uniform random sampling to find a good starting point
        n_uniform = 50
        best_x = None
        best_f = np.inf

        for i in range(n_uniform):
            if time.time() > deadline - 10:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES optimization
        # Initialize CMA-ES parameters
        x_mean = best_x.copy()
        f_mean = best_f
        sigma = 1.0  # Initial step size
        lambda_ = 4 * dim + 4  # Population size
        # CMA-ES weights and variance-effective population size
        mu = lambda_ // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        h_sigma = 0.0

        calls_made = n_uniform
        remaining = budget - calls_made

        for iteration in range(remaining):
            if time.time() > deadline - 5:
                break

            # Generate offspring
            offspring = []
            for i in range(lambda_):
                # Transform standard normal using Cholesky-like update
                z = rng.standard_normal(dim)
                # Simple approximation: use diagonal scaling (C is often near diagonal early)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            if time.time() > deadline - 2:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Update mean
            new_x_mean = np.zeros(dim)
            for i in range(mu):
                new_x_mean += weights[i] * offspring[i][1]
            
            # Update sigma (CSA)
            z_mean = C_inv_sqrt.T @ (new_x_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            p_sigma_norm = np.linalg.norm(p_sigma)
            
            h_sigma = 1.0 if (p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) / chi_n < (1.4 + 2 / (dim + 1)) * chi_n) else 0.0
            
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update evolution path and covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_x_mean - x_mean) / sigma
            C = (1 + c_1 * h_sigma * (h_sigma - 2 + c_c) + c_mu) * C
            for i in range(mu):
                diff = offspring[i][1] - x_mean
                C += c_mu * weights[i] * np.outer(diff, diff)

            # Update C_inv_sqrt (approximation via eigenvalue decomposition for small dim)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                pass

            x_mean = new_x_mean
            f_mean = offspring[0][0]

            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
