"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with per-dimension step sizes and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # CMA-ES implementation
        def cma_es(func, x0, sigma0, budget_cma, rng, deadline, dim):
            # CMA-ES parameters
            lambda_ = 4 * int(np.ceil(3 + np.log(dim)))  # population size
            # Learning rates
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            # Initialize
            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            best_x = x.copy()
            best_f = func(x)
            calls = 1

            for gen in range(budget_cma):
                if not time_ok():
                    break

                # Generate offspring
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    offspring.append((fy, y))

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()

                # Compute weighted mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * offspring[i][1]

                # Update evolution paths
                y_vec = (x_new - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) <
                           (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                outer = np.outer(p_c, p_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * np.eye(dim)
                # Approximate update (simplified for speed)
                for i in range(mu):
                    z_i = np.linalg.solve(B @ D, offspring[i][1] - x)
                    C += c_mu * weights[i] * np.outer(z_i, z_i)

                # Eigen decomposition
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-20))
                B = B2
                D = D2

                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) /
                                                     np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                x = x_new

                if calls >= budget_cma:
                    break

            return best_x, best_f, calls

        # Phase 1: CMA-ES with moderate budget
        x0 = np.zeros(dim)
        sigma0 = 2.0
        cma_budget = min(budget - 50, 350)
        best_x, best_f, calls_used = cma_es(func, x0, sigma0, cma_budget, rng, deadline, dim)

        # Phase 2: Local refinement with (1+1)-ES
        remaining = budget - calls_used
        if remaining > 10 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(remaining):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma = min(sigma * 1.15, 1.0)
                else:
                    sigma = max(sigma * 0.9, 1e-8)

        return best_x, best_f
# EVOLVE-BLOCK-END
