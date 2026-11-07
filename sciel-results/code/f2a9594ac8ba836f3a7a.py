"""CMA-ES optimizer for BBOB f6 (Attractive sector) with global-to-local strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with global exploration phase and adaptive step-size control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Global exploration - 50 random evaluations
        n_explore = 50
        for _ in range(n_explore):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            total_calls += 1
            if fx_try < best_f:
                best_f = fx_try
                best_x = x_try.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        # Phase 2: CMA-ES local optimization
        # Standard CMA-ES parameters
        lambda_pop = int(4 + 3 * np.log(dim))  # population size
        mu = lambda_pop // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Step size
        sigma = 1.0
        x_mean = best_x.copy()

        # Covariance matrix
        C = np.eye(dim)
        C_eigen = np.linalg.eigh(C)
        C_eigvals = C_eigen[0]
        C_eigvecs = C_eigen[1]

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track best
        best_x = x_mean.copy()
        best_f = best_f

        while total_calls < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lambda_pop):
                if total_calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Transform z using eigendecomposition of C
                z_transformed = C_eigvecs @ (C_eigvals ** 0.5 * z)
                y = x_mean + sigma * z_transformed
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                offspring.append((y, fy))

            if not offspring:
                break

            # Sort offspring
            offspring.sort(key=lambda t: t[1])

            # Update best
            if offspring[0][1] < best_f:
                best_f = offspring[0][1]
                best_x = offspring[0][0].copy()

            # Recombination: weighted average of top mu offspring
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * offspring[i][0]

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = y_w

            # Update evolution paths
            # Compute inverse square root of C
            C_sqrt_inv = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T

            diff = (x_mean - x_mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ diff
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Update C
            h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (total_calls - n_explore) / lambda_pop))) / chi_n < 1.4 + 2 / (dim + 1)
            h_sigma = float(h_sigma)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * np.outer(offspring[i][0] - x_mean_old, offspring[i][0] - x_mean_old) for i in range(mu))

            # Update eigen decomposition
            C_eigen = np.linalg.eigh(C)
            C_eigvals = np.maximum(C_eigen[0], 1e-20)
            C_eigvecs = C_eigen[1]

            # Adapt step size
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Check for stagnation
            if norm_p_sigma < 1e-10 and sigma < 1e-4:
                # Restart with a new random point
                x_new = rng.uniform(-5, 5, dim)
                fx_new = func(x_new)
                total_calls += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
                x_mean = x_new.copy()
                sigma = 1.0
                C = np.eye(dim)
                C_eigen = np.linalg.eigh(C)
                C_eigvals = C_eigen[0]
                C_eigvecs = C_eigen[1]
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
