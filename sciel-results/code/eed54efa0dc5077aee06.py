"""Rewritten program using a population-based CMA-ES for the Rastrigin rotated task."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based CMA-ES with lambda=10 for global diversity."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start at the center of the box
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers ~[-4, 4]

        # Population size
        # Larger lambda helps with global exploration for Rastrigin
        lambda_ = 20
        mu = lambda_ // 2  # Number of parents

        # Weights for recombination
        w = np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Initial evaluation
        x0 = mean.copy()
        fx = func(x0)
        best_x = x0.copy()
        best_f = fx
        calls = 1

        # Pre-compute constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Use a fixed seed for reproducibility, but different from 0 to avoid potential bad luck
        rng = np.random.default_rng(42)

        while calls < budget:
            # Determine how many individuals we can evaluate in this generation
            n_eval = min(lambda_, budget - calls)
            if n_eval == 0:
                break

            # Generate n_eval samples
            # Sample from N(mean, sigma^2 * C)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 0)
                L = eigvecs @ np.diag(np.sqrt(eigvals))

            Z = rng.standard_normal((n_eval, dim))
            X = mean + sigma * (Z @ L.T)

            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate
            new_xs = []
            new_fs = []
            for i in range(n_eval):
                x_i = X[i]
                f_i = func(x_i)
                new_xs.append(x_i)
                new_fs.append(f_i)
                calls += 1

                # Update best
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()

            # Sort by fitness
            idx = np.argsort(new_fs)
            new_xs = np.array(new_xs)[idx]
            new_fs = np.array(new_fs)[idx]

            # Update mean
            # Use only the first mu individuals for recombination
            # If n_eval < mu, adjust weights or use all available
            if n_eval < mu:
                w_curr = w[:n_eval]
                w_curr /= np.sum(w_curr)
                mu_eff_curr = 1.0 / np.sum(w_curr ** 2)
            else:
                w_curr = w
                mu_eff_curr = mu_eff

            y_w = np.sum(w_curr[:, np.newaxis] * (new_xs[:n_eval] - mean)[np.newaxis, :, :], axis=0) / sigma

            # Update path evolution strategies
            norm_y_w = np.linalg.norm(y_w)
            if norm_y_w > 1e-12:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff_curr) * (y_w / norm_y_w)
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff_curr) * y_w

            # Update covariance matrix
            # Simplified h_sigma: check if path length is within expected bounds
            # This is an approximation for the delta h term
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < 1.5 * chi_n else 0.0

            # Rank-one update
            C = (1 + c_1 * h_sigma * (mu_eff_curr - 1) - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T - (1 - h_sigma) * C)
            # Rank-mu update
            C += c_mu * np.sum(w_curr[:, np.newaxis, np.newaxis] * (new_xs[:n_eval] - mean)[:, np.newaxis, :] @ (new_xs[:n_eval] - mean)[:, :, np.newaxis], axis=0)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Ensure sigma doesn't become too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update mean
            mean = mean + sigma * y_w

            # Optional: Clip mean to box to keep it in bounds
            mean = np.clip(mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
