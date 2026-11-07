"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Standard CMA-ES settings
        mu = dim // 2 + 1  # number of parents
        lambda_ = 2 * mu   # number of offspring
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial sigma based on box size
        sigma = 2.0

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)

        # Covariance matrix and evolution path
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        # Best tracking
        best_x = x.copy()
        best_fx = fx

        # Population-based CMA-ES
        it = 0
        calls_made = 1

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate lambda_ offspring
            pop = []
            pop_f = []

            for i in range(lambda_):
                if time.time() > deadline:
                    break

                # Sample from N(x, sigma^2 * C)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: use identity
                    L = np.eye(dim)

                z = np.random.randn(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls_made += 1

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

                pop.append(y)
                pop_f.append(fy)

            if len(pop) < lambda_:
                break

            # Sort by fitness
            idx = np.argsort(pop_f)
            pop = [pop[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]

            # Update evolution paths
            # Weighted mean of selected parents
            m_new = np.zeros(dim)
            for i in range(mu):
                m_new += weights[i] * pop[i]

            # Update sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (L.T @ (m_new - x) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m_new - x) / sigma

            c_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) * (np.linalg.norm(ps) / chi_n - 1)
            sigma *= np.exp(cs / c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update covariance matrix
            delta_var = (1 - hsig) * cc * (2 - cc)
            outer = np.outer(pc, pc)
            C = (1 + c1 * delta_var - c1 - cmu) * C
            C += c1 * (outer - C)

            for i in range(mu):
                yi = pop[i] - m_new
                C += cmu * weights[i] * np.outer(yi, yi)

            # Ensure C is positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = (eigvecs * eigvals) @ eigvecs.T

            x = m_new

            it += 1

        # Final polish: local search around best point
        if calls_made < budget and time.time() < deadline - 5:
            # Try small perturbations around best_x
            local_sigma = 0.01
            for _ in range(min(50, budget - calls_made)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + local_sigma * np.random.randn(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
