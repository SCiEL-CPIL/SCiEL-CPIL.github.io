"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        x_mean = np.zeros(n)
        sigma = 2.0

        # Covariance matrix
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)  # Initially identity

        # Evolution path
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Strategy parameters
        mu = max(1, n // 2)  # Number of recombination weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / (n + 2)) / ((n + 2) ** 2 + mu_eff))

        # Bounds
        lb = -5.0
        ub = 5.0

        # Evaluate initial point
        x0 = np.clip(x_mean, lb, ub)
        fx0 = func(x0)
        best_x = x0.copy()
        best_f = fx0
        calls = 1

        # Precompute Cholesky decomposition of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(n)

        while calls < budget:
            # Generate offspring
            offspring = []
            for i in range(mu):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, lb, ub)
                offspring.append(y)

            # Evaluate offspring
            for y in offspring:
                if calls >= budget:
                    break
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if calls >= budget:
                break

            # Sort offspring by fitness
            offspring_with_f = [(func_y, y) for y in offspring]
            # We already evaluated them, so we need to re-evaluate or store
            # Let's restructure: evaluate and store

        # Actually, let me rewrite this more carefully
        # Restart the implementation

        # Reset
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        calls = 0
        best_x = np.zeros(n)
        best_f = float('inf')

        mu = max(1, n // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / (n + 2)) / ((n + 2) ** 2 + mu_eff))

        lb = -5.0
        ub = 5.0

        # Evaluate initial mean
        x0 = np.clip(x_mean, lb, ub)
        fx0 = func(x0)
        calls += 1
        best_x = x0.copy()
        best_f = fx0

        # Cholesky of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(n)

        while calls < budget:
            # Generate mu offspring
            offspring = []
            for i in range(mu):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, lb, ub)
                offspring.append(y)

            # Evaluate all offspring
            offspring_f = []
            for y in offspring:
                if calls >= budget:
                    break
                fy = func(y)
                calls += 1
                offspring_f.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if calls >= budget:
                break

            if len(offspring_f) < mu:
                # Not enough evaluations, pad with best
                while len(offspring_f) < mu:
                    offspring_f.append((best_f, best_x.copy()))

            # Sort by fitness
            offspring_f.sort(key=lambda x: x[0])

            # Update mean
            x_new = np.sum([w[i] * offspring_f[i][1] for i in range(mu)], axis=0)

            # Update step size control variable
            # Need inverse square root of C for ps update
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)

            # Update evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new - x_mean) / sigma)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update evolution path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma

            # Update covariance matrix
            v = (x_new - x_mean) / sigma
            C = (1 + c_1 + c_mu - c_1) * C \
                + c_1 * (np.outer(pc, pc) - C) \
                + c_mu * np.sum([w[i] * np.outer(offspring_f[i][1] - x_mean,
                                                  offspring_f[i][1] - x_mean) for i in range(mu)], axis=0) / sigma ** 2

            # Ensure C is positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)

            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
