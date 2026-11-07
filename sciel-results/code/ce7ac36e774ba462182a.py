"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with robust implementation for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initial population
        x0 = np.zeros(dim)
        sigma = 2.0

        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))  # ~13 for dim=5

        # Weights
        weights = np.log(lambda_pop / 2 + 0.5) - np.log(np.arange(1, lambda_pop + 1))
        weights /= np.sum(weights)
        mu = lambda_pop // 2
        mueff = int(np.sum(weights[:mu]))

        # Step size control
        c_sigma = (mueff + 0.3) / ((dim + 1.3) + mueff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / ((dim + 4) * (2 + mueff / dim))
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Hessian matrix
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        ps = 0.0

        def update_C_sqrt_inv(C, C_sqrt_inv):
            """Compute inverse square root of C."""
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            return inv_sqrt

        def update_C_sqrt(C, C_sqrt_inv):
            """Compute square root of C."""
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            return sqrt

        C_sqrt_inv = update_C_sqrt_inv(C, C_sqrt_inv)
        C_sqrt = update_C_sqrt(C, C_sqrt_inv)

        # Best solution tracking
        best_x = x0.copy()
        best_f = func(best_x)
        calls_made = 1

        # Main loop
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            y_list = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = x0 + sigma * C_sqrt_inv @ z
                y = np.clip(y, -5, 5)
                y_list.append(y)

            # Evaluate population
            f_list = [func(y) for y in y_list]
            calls_made += len(y_list)

            # Sort by fitness
            idx = np.argsort(f_list)
            y_sorted = [y_list[i] for i in idx]
            f_sorted = [f_list[i] for i in idx]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Weighted mean of top mu individuals
            x_new = np.sum([weights[i] * y_sorted[i] for i in range(mu)], axis=0)

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_sqrt_inv @ (x_new - x0) / sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update covariance matrix
            y_vecs = [y_sorted[i] - x0 for i in range(mu)]
            delta_x = (x_new - x0) / sigma
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - dim - 1) / lambda_pop))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0

            C = (1 + c_c * (c_1 + c_mu) - c_1 - c_mu) * C
            C += c_1 * (np.outer(delta_x, delta_x) - C)
            C += c_mu * np.sum([weights[i] * np.outer(y_vecs[i] / sigma, y_vecs[i] / sigma) for i in range(mu)], axis=0)

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * delta_x

            # Update square root matrices
            C_sqrt_inv = update_C_sqrt_inv(C, C_sqrt_inv)
            C_sqrt = update_C_sqrt(C, C_sqrt_inv)

            # Move mean
            x0 = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
