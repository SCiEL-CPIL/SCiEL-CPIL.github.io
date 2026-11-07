"""Optimizer for BBOB f4 (Büche-Rastrigin), 5D, 500 budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style population-based continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s
        dim = self.dim
        budget = self.budget

        # Initial point: center of the box (shifted minimum is in [-4,4]^5)
        x = np.zeros(dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls = 1

        # CMA-ES parameters
        pop_size = 10  # lambda
        # Initial sigma: large enough to explore, small enough for progress
        sigma = 1.0

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        c_sigma = 1.0 / (pop_size + 3)
        c_c = (2 + 2.0 / (dim + 1)) / ((dim + 2) + 2 + 2.0 / (dim + 1))
        d_sigma = 1 + 2 * max(0, (dim - 1) / (dim + 2) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Covariance matrix
        C = np.eye(dim)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        # Precompute inverse square root of C (initially identity)
        C_inv_sqrt = np.eye(dim)

        # Update interval for eigendecomposition
        update_interval = max(1, int(1 / (c_c * (4 + dim / dim))))
        step_count = 0

        # Ranking weights for CMA-ES
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = 1.0 / (pop_size + 3)
        c_c = (2 + 2.0 / (dim + 1)) / ((dim + 2) + 2 + 2.0 / (dim + 1))

        def sample_normals():
            """Generate pop_size x dim matrix of standard normals."""
            return rng.standard_normal((pop_size, dim))

        rng = np.random.default_rng(42)

        while calls < budget:
            # Time check
            if time.time() > deadline:
                break

            step_count += 1

            # Sample offspring
            # Use current C via eigendecomposition
            if step_count % update_interval == 0:
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.maximum(C_eigvals, 1e-16)
                C_inv_sqrt = (C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals))) @ C_eigvecs.T
                # Regularize
                C = np.maximum(C, 1e-16)

            normals = sample_normals()
            # Transform: x + sigma * C^{1/2} * z
            # C^{1/2} = eigvecs @ diag(sqrt(eigvals)) @ eigvecs.T
            C_sqrt = C_eigvecs @ np.diag(np.sqrt(np.maximum(C_eigvals, 1e-16))) @ C_eigvecs.T
            offspring = np.array([x + sigma * (C_sqrt @ z) for z in normals])
            # Clip to box
            offspring = np.clip(offspring, -5, 5)

            # Evaluate offspring
            offspring_f = np.array([func(offspring[i]) for i in range(pop_size)])
            calls += pop_size

            # Track best
            for i in range(pop_size):
                if offspring_f[i] < best_f:
                    best_f = offspring_f[i]
                    best_x = offspring[i].copy()

            # Sort by fitness
            order = np.argsort(offspring_f)
            sorted_offspring = offspring[order]
            sorted_f = offspring_f[order]

            # Weighted mean
            w_mean = np.zeros(dim)
            for i in range(mu):
                w_mean += weights[i] * sorted_offspring[i]
            y = w_mean - x  # step direction

            # Update evolution paths
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (y / sigma))

            # p_c update
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * step_count))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ y)

            # Update x
            x = w_mean

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update covariance matrix
            delta_d = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                zi = C_inv_sqrt @ sorted_offspring[i]
                rank_mu += weights[i] * np.outer(zi, zi)
            C = (1 + c_c * delta_d - c_c - c_c * mu_eff) * C + c_c * rank_one + c_c * mu_eff * rank_mu
            # Regularize
            C = (C + C.T) / 2
            C[np.diag_indices(dim)] += 1e-16

            # Early stopping: if sigma is very small and no improvement
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
