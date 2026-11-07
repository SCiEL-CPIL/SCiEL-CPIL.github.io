"""CMA-ES optimizer for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation with budget and time constraints."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Weights
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1) + 0.5)
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Initial sigma
        sigma = 2.0

        # Initial center: use a point that might be closer to optimum
        # Since the minimizer is in [-4, 4]^5, start near origin
        x_mean = np.zeros(dim)
        fx_mean = func(x_mean)
        best_x = x_mean.copy()
        best_fx = fx_mean

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)

        # Step size control parameters
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        c_sigma = (mu_eff + 2) / (dim + 5)
        c_cov = (2 + 2 * min(weights[0], 0.5 / mu_eff) - mu_eff / dim) / \
                ((dim + 2) ** 2 + mu_eff)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Pre-compute Cholesky-like factor for sampling
        # We need sqrt(eigvals) * eigvecs
        D = np.sqrt(np.maximum(eigvals, 1e-20))
        B = eigvecs * D  # B = eigvecs @ diag(sqrt(eigvals))

        calls_made = 1

        for iteration in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            # Sample population
            # Generate standard normal samples
            z = rng.standard_normal((lam, dim))
            # Transform to current distribution
            # y = x_mean + sigma * z @ B.T
            population = x_mean + sigma * (z @ B.T)

            # Clip to box
            population = np.clip(population, -5, 5)

            # Evaluate population
            fvals = np.array([func(x) for x in population])
            calls_made += lam

            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = population[order]
            f_sorted = fvals[order]

            # Update best
            if f_sorted[0] < best_fx:
                best_fx = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Update mean
            x_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update evolution paths
            # Inverse square root of C for p_sigma
            # C^{-1/2} = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            inv_sqrt_C = (eigvecs * (1.0 / np.maximum(D, 1e-20))) @ eigvecs.T

            y_inv = inv_sqrt_C @ (x_new - x_mean)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_inv
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mu_eff) * (x_new - x_mean) / sigma

            # Update sigma
            sigma *= np.exp(c_sigma / ds * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-15, 1e15)

            # Update covariance matrix
            delta_h = (1 + 3 * cs / (1 - cs)) * \
                      (np.linalg.norm(p_c) ** 2 / dim - 1)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None, None] *
                             np.outer(y_inv - x_mean / sigma,
                                      y_inv - x_mean / sigma), axis=0)
            C = (1 + c_cov * (mu_eff - 2 + delta_h) - c_cov) * C + \
                c_cov * (rank_one + rank_mu)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization to ensure positive definiteness
            C += 1e-12 * np.eye(dim)

            # Recompute eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs * D
            except np.linalg.LinAlgError:
                # Fallback: reset covariance
                C = np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(eigvals, 1e-20))
                B = eigvecs * D

            x_mean = x_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
