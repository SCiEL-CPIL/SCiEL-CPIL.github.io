"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Schwefel f20 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        n = 8  # population size
        mu = n // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)

        c_sigma = 0.8 / (dim + 3)
        c_cov = 2 * (2 - 0.5) / ((dim + 1.3)**2 + 0.5)
        c_c = (4 + dim / 4) / (dim + 4.1)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Initialize
        x = rng.uniform(-5, 5, dim)
        f_best = func(x)
        x_best = x.copy()
        sigma = 3.0
        C = np.eye(dim)
        L = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls = 1

        # Generate initial population
        pop = np.zeros((n, dim))
        fvals = np.zeros(n)
        for i in range(n):
            z = rng.standard_normal(dim)
            pop[i] = np.clip(x + sigma * (L @ z), -5, 5)
            fvals[i] = func(pop[i])
            calls += 1
            if fvals[i] < f_best:
                f_best = fvals[i]
                x_best = pop[i].copy()

        order = np.argsort(fvals)
        pop = pop[order]
        fvals = fvals[order]

        while calls < budget:
            if time.time() > deadline:
                break

            # Recombination mean
            m = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Path updates
            z_mean = np.linalg.solve(L, m - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(calls+1))) < (1.4 + 2/(dim+1)) * chi_n) else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma

            # Covariance update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = (c_mu := 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff)) * \
                      (weights[:, None] * weights[None, :]) @ np.outer(m - x, m - x) / sigma**2
            C = (1 + c_cov * (c_c * (2 - c_c) + delta_h) - c_cov - c_mu) * C \
                + c_cov * np.outer(p_c, p_c) + rank_mu
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

            # Sigma update
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # New population
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                pop[i] = np.clip(m + sigma * (L @ z), -5, 5)
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < f_best:
                    f_best = fvals[i]
                    x_best = pop[i].copy()

            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Update center
            x = m

        return x_best, f_best
# EVOLVE-BLOCK-END
