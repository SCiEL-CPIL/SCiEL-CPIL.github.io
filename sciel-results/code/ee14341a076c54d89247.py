"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with CMA-ES-like adaptation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # Start at origin (better prior for shifted minimizer in [-4,4])
        x = np.zeros(dim)
        fx = func(x)

        # Step size with adaptive scaling
        sigma = 1.5

        # Covariance matrix initialization
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Parameters
        c_sigma = (1 + 2 / max(1, dim - 2)) / (dim + 3)
        c_c = (4 + dim / 3) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * (dim * 0.35 + 0.5) / ((dim + 2) ** 2 + dim))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        d_sigma = c_sigma * (1 + 2 / max(1, dim - 2) + c_mu / c_sigma)
        cs = (c_c + 2) / (dim + c_c + 2)
        ds = 1 + 2 * max(0, np.sqrt((dim + 1) / 3 - 1) - 1) + c_c

        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))

        for gen in range(self.budget // lambda_ + 1):
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lambda_):
                # Whitening transformation
                z = rng.standard_normal(dim)
                # Apply C^(1/2) using Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    y = z
                x_off = x + sigma * y
                x_off = np.clip(x_off, -5, 5)
                f_off = func(x_off)
                offspring.append((f_off, x_off))

            if time.time() > deadline:
                break

            # Sort offspring
            offspring.sort(key=lambda t: t[0])

            # Selection
            mu = max(1, lambda_ // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Update mean
            x_new = sum(w[i] * offspring[i][1] for i in range(mu))

            # Update covariance
            y_w = np.array([offspring[i][1] for i in range(mu)])
            y_w -= x
            y_w /= sigma

            # Rank-mu update
            delta_c = (c_c / (1 - c_c)) * (1 - np.exp(-ds))
            C += -c_1 * (1 + delta_c) * np.outer(x - x_new, x - x_new) / sigma ** 2
            C += c_1 * np.sum(w[i] * np.outer(y_w[i], y_w[i]) for i in range(mu))
            C += c_mu * np.sum(w[i] * np.outer(y_w[i], y_w[i]) for i in range(mu))

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_new - x)
            sigma *= np.exp(d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update correlation matrix path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_new - x)
            h_sig = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            C += -c_1 * (1 + delta_c) * np.outer(x - x_new, x - x_new) / sigma ** 2 + c_1 * (1 - h_sig) * c_c * (2 - c_c) * (np.outer(p_c, p_c) - C)

            x = x_new
            fx = min(fx, offspring[0][0])

        return x, fx
# EVOLVE-BLOCK-END
