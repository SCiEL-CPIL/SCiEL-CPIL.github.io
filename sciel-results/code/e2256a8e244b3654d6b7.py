"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))
        # Step size initialization
        sigma = 2.0
        # Starting point: center of the box
        x = np.zeros(dim)
        # Covariance matrix: identity
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Learning rates
        mu = lam // 2  # Number of best individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Sample population
            B = np.random.randn(lam, dim)
            # Whitening: B * C^(1/2)
            # Compute eigenvalue decomposition for C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            # Actually, for efficiency, we can use Cholesky or just sample from N(0, C)
            # Let's use: y = x + sigma * (C^(1/2) @ z) where z ~ N(0, I)
            # But C^(1/2) is expensive. Alternative: use the fact that we can sample
            # directly using the eigen decomposition.
            
            # Simpler approach: sample z ~ N(0, I), then y = x + sigma * (eigvecs @ (sqrt(eigvals) * z))
            # This is equivalent to sampling from N(0, C)
            
            # Sample population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                # y = x + sigma * (C^(1/2) @ z)
                # C^(1/2) @ z = eigvecs @ (sqrt(eigvals) * (eigvecs.T @ z))
                proj = eigvecs.T @ z
                scaled = np.sqrt(eigvals) * proj
                C_half_z = eigvecs @ scaled
                Y[i] = x + sigma * C_half_z
                Y[i] = np.clip(Y[i], -5, 5)

            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                F[i] = func(Y[i])
                calls += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = Y[i].copy()

            # If we didn't evaluate all, pad with inf
            if calls < budget and time.time() <= deadline:
                for i in range(int(np.sum(np.isfinite(F))), lam):
                    F[i] = np.inf

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * Y_sorted[i]
            
            # Update evolution paths
            y_diff = x_new - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ np.linalg.inv(C_half_z_matrix)) @ (y_diff / sigma) if hasattr(self, 'C_half_z_matrix') else (1 - c_sigma) * p_sigma
            
            # This is getting complicated. Let me use a simpler CMA-ES implementation.
            # Actually, let me restart with a cleaner implementation.
            break  # Break to restart with cleaner code

        return best_x, best_fx


# Let me rewrite this more carefully


class Optimizer:
    """CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # Population size
        mu = lam // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initialize
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Precompute C^(1/2) using Cholesky (C is PSD)
        # We'll update this as C changes

        def sample_from_C(C_mat, n_samples=1):
            """Sample n_samples vectors from N(0, C_mat)"""
            # Use eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.maximum(eigvals, 1e-10)
            if n_samples == 1:
                z = np.random.randn(dim)
                proj = eigvecs.T @ z
                scaled = np.sqrt(eigvals) * proj
                return eigvecs @ scaled
            else:
                Z = np.random.randn(n_samples, dim)
                proj = Z @ eigvecs
                scaled = proj * np.sqrt(eigvals)[None, :]
                return scaled @ eigvecs.T

        while calls < budget:
            if time.time() > deadline:
                break

            # Sample population
            Z = np.random.randn(lam, dim)
            # Sample from N(0, C)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            proj = Z @ eigvecs
            scaled = proj * np.sqrt(eigvals)[None, :]
            Y = x[None, :] + sigma * (scaled @ eigvecs.T)
            Y = np.clip(Y, -5, 5)

            # Evaluate
            F = np.full(lam, np.inf)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                F[i] = func(Y[i])
                calls += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = Y[i].copy()

            # Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * Y_sorted[i]

            # Update evolution paths
            y_diff = x_new - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_diff / sigma)

            # Update C
            H = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (Y_sorted[i] - x) / sigma
                H += weights[i] * np.outer(y_i, y_i)

            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - dim - 1)))) / np.sqrt(1 - (1 - c_sigma) ** 2) if calls > dim + 1 else 1.0
            h_sigma = min(h_sigma, 1.4)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_diff / sigma)

            C = (1 + c_1 + c_mu - c_1 - mu_eff * c_mu) * C + \
                c_1 * (np.outer(p_c, p_c) - C) + \
                c_mu * H

            # Ensure C is symmetric and PSD
            C = (C + C.T) / 2
            # Eigen decomposition to ensure PSD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = x_new

            # Check for convergence
            if np.all(np.abs(x_new - x) < 1e-8):
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
