"""Rewritten program using a CMA-ES style approach with adaptive step sizes."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for the BBOB-style f13 (Sharp ridge) problem."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        n = self.budget

        # Initialize center and covariance
        x_mean = np.zeros(d)
        C = np.eye(d)
        sigma = 2.0

        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, lambda_ + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Evolution path
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma

        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)
        sqrt_inv_eig = 1.0 / np.sqrt(np.maximum(eigvals, 1e-12))
        # Precompute sqrt(C) via eigvecs @ diag(sqrt(eigvals)) @ eigvecs.T
        sqrt_C = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T

        best_x = x_mean.copy()
        best_f = float('inf')
        calls = 0

        for gen in range(n):
            # Generate population
            Y = np.zeros((lambda_, d))
            for i in range(lambda_):
                # Sample from N(0, C) using eigendecomposition
                z = rng.standard_normal(d)
                # Apply sqrt(C)
                Y[i] = sqrt_C @ z

            # Clip to box
            X = x_mean[None, :] + sigma * Y
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= n:
                    break
                F[i] = func(X[i])
                calls += 1

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Compute new mean
            x_new_mean = np.zeros(d)
            for i in range(mu):
                x_new_mean += weights[i] * X_sorted[i]

            # Normalize step
            step = x_new_mean - x_mean
            step_norm = np.linalg.norm(step)
            if step_norm > 1e-16:
                step /= step_norm
            else:
                step = np.zeros(d)

            # Update sigma
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * step
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) / chi_n < (1.4 + 2 / (d + 1))
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            delta_h = (1 - c_c) ** (2 * (gen + 1))
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                y_i = (x_new_mean - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            C = (1 + c_1 * (c_mu + c_1 - 1) + c_mu * (1 - delta_h) - c_1) * C \
                + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive semi-definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_mean = x_new_mean

            # Recompute sqrt(C) for next generation
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # If we've used all budget, break
            if calls >= n:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
