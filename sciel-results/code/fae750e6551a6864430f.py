"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 10
        mu = lam // 2

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        # Step size
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Learning rates
        c_sigma = (mu + 2) / (dim + 4)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        w = w[:mu]
        mu_eff = 1.0 / np.sum(w ** 2)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))

        # Damping
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Main loop
        while call_count < budget:
            if time.time() > deadline:
                break

            # Generate and evaluate population
            population = []
            for i in range(lam):
                if call_count >= budget:
                    break
                # Sample from N(x, sigma^2 * C) using Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    z = rng.standard_normal(dim)
                    y = x + sigma * (L @ z)
                except:
                    y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                population.append((y, fy))

            if len(population) < mu:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[1])
            selected = population[:mu]

            # Weighted mean
            mean_y = np.zeros(dim)
            for i in range(mu):
                mean_y += w[i] * selected[i][0]

            # Update evolution paths
            diff = (mean_y - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            h_sig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * len(population))) / chi_n
            if h_sig < (1 + 3 / dim) * (1 - c_c):
                p_c = (1 - c_c) * p_c + c_c * diff

            # Update mean
            x = mean_y

            # Update covariance
            yw = [selected[i][0] - x for i in range(mu)]
            C = (1 + c_1 * (h_sig ** 2 - 1) * c_c * (2 - c_c)) * C
            C -= c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C += c_mu * w[i] * np.outer(yw[i], yw[i])

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += 1e-10 * np.eye(dim)
            except:
                C = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_fx

# EVOLVE-BLOCK-END
