"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rank-1 + Rank-mu natural gradient descent with diagonal step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Population size
        lam = 10
        mu = lam // 2

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(dim)  # start at center
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x

        # Diagonal step sizes (log scale for stability)
        sigma = np.ones(dim) * 2.0

        # Cumulative step size adaptation
        cs = 1.0
        c_sigma = (mu + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu + 1) / (dim + 1)) - 1)
        chi_mu = np.sqrt(mu) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Path evolution strategies
        c_1 = (2 * (mu_eff - 2 + 1 / dim)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))
        mu_eff = 1.0 / np.sum(1.0 / np.arange(1, mu + 1))
        c_1 = (2 * (mu_eff - 2 + 1 / dim)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))

        # Rank-1 and Rank-mu weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)
        c_1 = (2 * (mu_eff - 2 + 1 / dim)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))

        # State variables for path evolution
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)

        calls_made = 1

        for iteration in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.zeros((lam, dim))
            f_y = np.zeros(lam)
            for i in range(lam):
                y[i] = x + sigma * rng.standard_normal(dim)
                y[i] = np.clip(y[i], -5, 5)
                f_y[i] = func(y[i])
                calls_made += 1
                if time.time() > deadline:
                    break

            if calls_made >= budget:
                break

            # Sort by fitness
            idx = np.argsort(f_y)
            y_sorted = y[idx]
            f_sorted = f_y[idx]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Weighted mean of best mu individuals
            y_w = np.sum(w[:, None] * y_sorted[:mu], axis=0)

            # Update step size
            sigma_old = sigma.copy()
            s = y_w - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(sigma) * s
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_mu - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update position
            x = y_w

            # Update covariance paths (diagonal approximation)
            p_c = (1 - c_1) * p_c + np.sqrt(c_1 * (2 - c_1) * mu_eff) / np.linalg.norm(s / sigma) * (s / sigma)
            p_s = (1 - c_mu) * p_s + np.sqrt(c_mu * (2 - c_mu) * mu_eff) / np.linalg.norm(y_sorted[:mu, 0] / sigma[0]) * ((y_sorted[:mu, 0] - x) / sigma)

            # Update diagonal step sizes using rank-1 and rank-mu updates
            # Rank-1 update
            sigma *= np.exp(0.5 * c_1 * (np.outer(p_c, p_c) / np.ones(dim) - 1) / dim)
            # Rank-mu update
            for i in range(mu):
                sigma *= np.exp(0.5 * c_mu * w[i] * ((y_sorted[i] - x) ** 2 / sigma ** 2 - 1) / dim)

            # Clip sigmas
            sigma = np.clip(sigma, 1e-10, 1e10)

            # If we've used all budget, stop
            if calls_made >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
