"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s kill

        # Initial point: center of the box is a good prior for BBOB-style
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        # Standard CMA-ES settings for dim=5
        lambda_ = 4 * int(np.ceil(3.0 / dim))  # population size
        # For dim=5: lambda = 4 * ceil(3/5) = 4 * 1 = 4
        # Let's use a larger population for better exploration
        lambda_ = max(4, 4 * int(np.ceil(3.0 / dim)))
        # Actually, let's use a standard setting: mu = lambda/2
        mu = lambda_ // 2

        # Weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Step size initialization
        sigma = 2.0

        # Evolution path parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 1.3) + c_c) / ((dim + 2) * (2 - c_c) + mueff / (dim + 1.3))
        c_mu = (2 - c_1 + mueff / (dim + 1.3) - c_c) / ((dim + 2) * (2 - c_c - c_1) + mueff / (dim + 1.3))

        # Matrices
        C = np.eye(dim)
        Cinv = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        Cinv_sqrt_inv = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial mean
        m = x.copy()

        # Track number of function evaluations
        n_evals = 1

        # Main loop
        while n_evals < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            offspring_f = []

            for i in range(lambda_):
                # Generate step
                z = np.random.randn(dim)
                # Apply CMA-ES transformation
                # y = m + sigma * Cinv_sqrt @ z
                y = m + sigma * (Cinv_sqrt_inv @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                fy = func(y)
                n_evals += 1
                offspring.append(y)
                offspring_f.append(fy)

                # Update best
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            # Sort offspring by fitness
            idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in idx]
            sorted_offspring_f = [offspring_f[i] for i in idx]

            # Compute weighted mean of best mu offspring
            new_m = np.zeros(dim)
            for i in range(mu):
                new_m += w[i] * sorted_offspring[i]

            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma
            if sigma > 1e-12:
                p_sigma += np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (Cinv_sqrt @ (new_m - m) / sigma)

            # Check if p_sigma is too large (chi distribution)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) - 1.0 / (21 * dim ** 2))
            if np.linalg.norm(p_sigma) > 1.4 + 2.0 / (dim + 1) * chi_n:
                c_sigma = max(c_sigma / 1.5, 1e-12)

            # p_c
            h_sigma = 1.0
            # Heuristic: check if p_sigma and p_c are aligned
            if np.linalg.norm(p_sigma) > 0:
                h_sigma = np.dot(p_sigma, Cinv_sqrt @ (new_m - m)) / (sigma * np.linalg.norm(p_sigma)) > (1 - c_c / 2) * chi_n

            p_c = (1 - c_c) * p_c
            if h_sigma:
                p_c += np.sqrt(c_c * (2 - c_c) * mueff) * (new_m - m) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            delta_h = 1 + delta_h
            C = (1 - c_1 - c_mu + delta_h * c_1) * C
            C += c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = (sorted_offspring[i] - m) / sigma
                C += c_mu * w[i] * np.outer(y_i, y_i)

            # Ensure positive semi-definite
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update Cinv_sqrt
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            Cinv_sqrt_inv = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Update mean
            m = new_m

            # If we've converged, we can stop early
            if sigma < 1e-10:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
