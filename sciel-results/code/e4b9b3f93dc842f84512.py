"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with population-based exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)

        # Initialize
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size
        sigma = 2.0

        # Covariance matrix (start as identity)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Chi approximation
        chi_mu = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls_made = 1
        generation = 0

        while calls_made < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lambda_, dim))
            offspring_fx = np.zeros(lambda_)

            for i in range(lambda_):
                if time.time() > deadline:
                    break
                # Sample from multivariate normal using current C
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                offspring[i] = y
                offspring_fx[i] = func(y)
                calls_made += 1

            if calls_made >= budget:
                break

            # Sort by fitness
            order = np.argsort(offspring_fx)
            sorted_offspring = offspring[order]
            sorted_fx = offspring_fx[order]

            # Update best
            if sorted_fx[0] < best_fx:
                best_fx = sorted_fx[0]
                best_x = sorted_offspring[0].copy()

            # Mean update
            x_new = sum(weights[i] * sorted_offspring[i] for i in range(mu))

            # Evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ (x_new - x))
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (C_sqrt @ (x_new - x))

            # Step size update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_mu - 1))

            # Covariance matrix update
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1)))) < \
                      (1.4 + 2 / (dim + 1)) * chi_mu
            h_sigma = float(h_sigma)

            rank_c = np.outer(p_c, p_c)
            rank_mu = sum(weights[i] * np.outer(C_sqrt @ (sorted_offspring[i] - x),
                                               C_sqrt @ (sorted_offspring[i] - x)) for i in range(mu))

            C = (1 + c_1 * (h_sigma * (dim + 1) - 1) - c_1 - c_mu) * C \
                + c_1 * (rank_c - (1 - h_sigma) * (1 - c_c) * C) \
                + c_mu * rank_mu

            # Ensure positive definite
            C = 0.5 * (C + C.T)
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: add small regularization
                C += 1e-6 * np.eye(dim)
                C = 0.5 * (C + C.T)
                try:
                    C_inv = np.linalg.inv(C)
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)

            x = x_new
            fx = best_fx  # track best for reference
            generation += 1

        # Final polish: local refinement around best_x
        if calls_made < budget and time.time() < deadline:
            sigma_polish = 0.01
            x_polish = best_x.copy()
            fx_polish = best_fx
            remaining = budget - calls_made
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x_polish + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx_polish:
                    x_polish = y
                    fx_polish = fy
                    sigma_polish *= 1.1
                else:
                    sigma_polish *= 0.85
                    if sigma_polish < 1e-6:
                        break
            if fx_polish < best_fx:
                best_fx = fx_polish
                best_x = x_polish

        return best_x, best_fx
# EVOLVE-BLOCK-END
