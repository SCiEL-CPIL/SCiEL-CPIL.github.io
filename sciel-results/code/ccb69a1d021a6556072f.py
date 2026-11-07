"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 325.0  # Safety margin before 330s kill

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        mu = lam // 2  # number of parents
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        # Step size control
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialization
        x = np.zeros(dim)
        f_x = func(x)
        sigma = 1.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        count_steps = 0

        best_x = x.copy()
        best_f = f_x

        # Track best so far for AOCC
        best_so_far = f_x

        for iteration in range(budget - 1):
            # Deadline check
            if time.time() > self.deadline:
                break

            # Generate offspring
            ps = np.linalg.cholesky(C)
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * ps @ z
                y = np.clip(y, -5, 5)
                f_y = func(y)
                offspring.append((y, f_y))
                if f_y < best_so_far:
                    best_so_far = f_y
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])

            # Update mean
            y_w = np.zeros((mu, dim))
            for i in range(mu):
                y_w[i] = (offspring[i][0] - x) / sigma

            x_new = x + sigma * (w @ y_w.T).T

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ (w @ y_w.T).T).T
            norm_ps = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1 + 1.4 / dim) * chi_n else 0.0
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            y_w_std = y_w / np.sqrt(mu_eff)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            C = (1 + c_1 + c_mu - c_c * (1 - h_sigma)) * C + \
                c_c * (p_c @ p_c.T - C) + \
                c_mu * (y_w_std.T @ (y_w_std * w[:, np.newaxis]).T)

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = x_new
            f_x = best_f  # Use best known value for tracking
            C_sqrt = np.linalg.cholesky(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
