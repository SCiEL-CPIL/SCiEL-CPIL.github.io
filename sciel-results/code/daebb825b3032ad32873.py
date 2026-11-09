"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size for BBOB f13 (sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize
        x = rng.uniform(-4.0, 4.0, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + 2) - 1))
        mu_eff = lambda_ / 2
        mu_w = np.arange(1, mu_eff + 1) ** np.log(2)
        mu_w /= mu_w.sum()
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls_made = 1

        for iteration in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            Y = rng.standard_normal((lambda_, dim))
            X = x + sigma * (Y @ B.T)
            X = np.clip(X, -5.0, 5.0)

            # Evaluate
            F = np.array([func(xi) for xi in X])
            calls_made += lambda_

            # Sort
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]

            # Update best
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Mean
            m_new = sum(mu_w[i] * X_sorted[i] for i in range(mu_eff))

            # Update path for sigma
            y_sigma = (m_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Update path for C
            if np.linalg.norm(y_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 1 / dim) * chi_n:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma

            # Update covariance
            delta_h = (1 - c_c) - c_c * (1 - c_c)
            C = (1 + c_1 + c_mu - c_1 - c_c) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * sum(mu_w[i] * np.outer(X_sorted[i] - x, X_sorted[i] - x) for i in range(mu_eff))
            C += c_c * delta_h * (np.outer(p_c, p_c) - C)

            # Symmetrize
            C = (C + C.T) / 2

            # Update mean
            x = m_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
