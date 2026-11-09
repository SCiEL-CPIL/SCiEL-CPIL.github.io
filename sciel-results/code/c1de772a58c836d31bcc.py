"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Step size
        sigma = 1.0
        # Mean
        x = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        # Eigenvalues of C
        D = np.ones(dim)
        # Inverse square root of C
        CS_inv = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialize
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1

        # Initial phase: use larger sigma to explore
        for i in range(3):
            sigma = 2.0
            break

        while calls < budget:
            # Sample population
            ys = np.zeros((lam, dim))
            fs = np.zeros(lam)
            for i in range(lam):
                ys[i] = x + sigma * (CS_inv @ rng.standard_normal(dim))
                ys[i] = np.clip(ys[i], -5, 5)
                fs[i] = func(ys[i])
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()

            # Weighted mean
            y_w = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

            # Update evolution paths
            # ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (CS_inv @ y_w)
            # h_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim) + 1 / (12 * dim ** 3))
            h_sigma = np.sqrt((1 - (1 - c_sigma) ** (2 * (budget - calls))) /
                              (1 - (1 - c_sigma) ** 2)) * \
                      (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * calls)) / chi_n - 1 +
                       2 / (dim + 1)) if calls > 0 else 1.0
            h_sigma = min(1.0, h_sigma)

            # pc
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (CS_inv @ y_w)

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + \
                c_1 * np.outer(pc, pc) + \
                c_mu * np.sum(weights[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x)
                               for i in range(mu))

            # Update eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.clip(D, 1e-20, None)
            # Ensure positive definite
            D = np.maximum(D, 1e-20)
            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = x + sigma * (B @ np.diag(D ** 0.5) @ (B.T @ y_w))

            # Recompute CS_inv
            D_sqrt_inv = 1.0 / np.sqrt(D)
            CS_inv = (B @ np.diag(D_sqrt_inv) @ B.T)

        return best_x, best_f
# EVOLVE-BLOCK-END
