"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        d = self.dim

        # Initialize with multiple random starting points, pick best
        n_init = 10
        best_x = None
        best_f = np.inf
        for _ in range(n_init):
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        calls_used = n_init
        remaining = self.budget - calls_used

        # CMA-ES-like parameters
        sigma = 2.0
        c1 = 1 / (d + 4)
        cmu = 2 / ((d + 2) ** 2 + 10)
        c_sigma = (d + 1.7) / (d + 9)
        d_sigma = 1 + 2 * max(0, np.sqrt((d - 1) / (d + 1)) - 1)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Covariance matrix
        C = np.eye(d)
        C_inv = np.eye(d)

        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Population size
        lambda_ = 4 + int(np.floor(3 * np.log(d)))  # ~11 for d=5
        mu = lambda_ // 2  # selection size

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)

        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f

        while calls_used < self.budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                # Sample from N(best_x, sigma^2 * C)
                # Simple Cholesky-free sampling using eigendecomposition occasionally
                # For speed, use simple Gaussian with sigma scaling
                y = best_x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append((fy, y))

                if time.time() > deadline:
                    break

            if calls_used >= self.budget or time.time() > deadline:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < global_best_f:
                global_best_f = offspring[0][0]
                global_best_x = offspring[0][1].copy()

            # Compute weighted mean of top mu
            x_mean = np.zeros(d)
            for i in range(min(mu, len(offspring))):
                x_mean += weights[i] * offspring[i][1]

            # Update covariance matrix (simplified: use rank-1 update)
            diff = x_mean - best_x
            C = (1 - c1 - cmu) * C + c1 * np.outer(diff, diff) + cmu * np.outer(diff, diff)
            # Ensure positive semi-definite (simple regularization)
            C = 0.5 * (C + C.T) + 1e-6 * np.eye(d)

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (diff / sigma)
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Update center
            best_x = x_mean

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
