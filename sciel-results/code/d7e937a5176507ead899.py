"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based optimizer for BBOB-style f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        # Initial mean
        x_mean = np.zeros(n)
        # Initial step size
        sigma = 2.0
        # Covariance matrix
        C = np.eye(n)
        # Eigen decomposition cache
        D = np.linalg.cholesky(C)
        
        # Learning rates
        c_sigma = (lambda_ / 2 + 1) / ((n + 1.3) + lambda_ / 2)
        c_c = (4 + lambda_ / n) / (n + 4 + 2 * lambda_ / n)
        c_1 = 2 / ((n + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 + 1) - 2) / ((n + 2) ** 2 + lambda_))
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation
        best_f = evaluate(x_mean)

        # Main loop
        while calls < budget:
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                x_new = x_mean + sigma * (D @ z)
                f_new = evaluate(x_new)
                offspring.append(x_new)
                offspring_f.append(f_new)
            
            if not offspring:
                break

            # Sort by fitness
            order = np.argsort(offspring_f)
            offspring = [offspring[i] for i in order]
            offspring_f = [offspring_f[i] for i in order]

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum([w[i] * offspring[i] for i in range(mu)], axis=0)
            
            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (D.T @ (x_mean - x_old)) / sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)  # Cap to avoid explosion

            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma
            
            # Rank-one update
            y = (x_mean - x_old) / sigma
            C += c_1 * (np.outer(y, y) + (1 - (1 - c_c) ** 2) / c_1 * np.eye(n) - C)
            
            # Rank-mu update
            for i in range(mu):
                y_i = (offspring[i] - x_old) / sigma
                C += c_mu * w[i] * np.outer(y_i, y_i)

            # Ensure positive semi-definite
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small diagonal if not PD
                C += 1e-10 * np.eye(n)
                D = np.linalg.cholesky(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
