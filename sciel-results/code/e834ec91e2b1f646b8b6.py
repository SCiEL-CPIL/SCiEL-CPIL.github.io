"""CMA-ES optimizer for BBOB-style Discus function."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = 4 + int(3 * np.log(dim))  # 10 for dim=5
        mu = lam // 2  # 5

        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Initial parameters
        x = np.zeros(dim)
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        D = 1.0
        sqrtC = np.eye(dim)

        # Cumulation paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 2 + mu + 0.5)
        d_sigma = 1 + 2 * max(0, (mu - 1) / (dim + 2) - 1) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu + 2) / (dim + 2) - 1) / ((dim + 2) ** 2 + mu))

        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Evaluate initial point
        x_clipped = np.clip(x, -5, 5)
        best_val = func(x_clipped)
        x_best = x_clipped.copy()
        calls = 1

        while calls < budget:
            # Generate population
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (D * (sqrtC @ z))
                ys[i] = np.clip(y, -5, 5)

            # Evaluate population
            f_values = np.zeros(lam)
            for i in range(lam):
                if calls >= budget:
                    f_values[i] = np.inf
                    continue
                f_values[i] = func(ys[i])
                calls += 1

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(f_values)
            ys_sorted = ys[order]
            f_sorted = f_values[order]

            # Update best
            if f_sorted[0] < best_val:
                best_val = f_sorted[0]
                x_best = ys_sorted[0].copy()

            # Mean recombination
            x_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

            # Cumulation path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (D * (sqrtC @ ((x_new - x) / sigma)))

            # Cumulation path for C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (D * (sqrtC @ ((x_new - x) / sigma)))

            # Update covariance matrix using rank-mu and rank-1 updates
            # Rank-mu update
            rank_mu = np.sum(weights[:, None, None] * np.outer(ys_sorted[:mu] - x, ys_sorted[:mu] - x) / sigma**2, axis=0)
            
            # Rank-1 update
            rank_1 = np.outer(p_c, p_c)
            
            # Combine
            C = (1 - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                C = (C + C.T) / 2
                L = np.linalg.cholesky(C)
                sqrtC = L.T
                C_inv = np.linalg.inv(C)
                D = 1.0 / np.sqrt(np.mean(np.diag(C_inv)))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                sqrtC = np.eye(dim)
                C_inv = np.eye(dim)
                D = 1.0

            x = x_new

            # Sigma update
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

        return x_best, best_val
# EVOLVE-BLOCK-END
