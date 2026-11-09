"""Optimizer for BBOB f21 (Gallagher 101 peaks) using CMA-ES with time guard."""
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        # Population size: 4 + floor(3 * ln(dim)) for dim=5 -> 4 + 3*1.609 = 8
        lambda_pop = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        # Step size: start relatively large to explore
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for efficient sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        invsqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation rates
        mu = lambda_pop // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma = c_sigma * (1 + min(1, c_sigma) - c_c)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma

        calls_made = 1

        for iteration in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Sample offspring
            offspring = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                # Transform via covariance
                y = x + sigma * (eigvecs @ (np.sqrt(eigvals) * (eigvecs.T @ z)))
                # Actually, let's use the proper transformation
                # x_new = x + sigma * B * D * z where C = B D^2 B^T
                y = x + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            # Check deadline mid-population
            if time.time() > deadline:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Selection: take top mu
            selected = [offspring[i][1] for i in range(mu)]
            selected_f = [offspring[i][0] for i in range(mu)]

            # Weighted mean of selected
            mean_y = np.sum([weights[i] * selected[i] for i in range(mu)], axis=0)

            # Update evolution paths
            diff = (mean_y - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            p_c = (1 - c_c) * p_c + c_c * diff

            # Check if p_sigma is too large (step size adaptation)
            hs = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lambda_pop + 1)))
            if hs < 4.4 + 2 / dim:
                sigma *= np.exp(c_sigma / d_sigma * (hs / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 50)

            # Update covariance matrix
            v = p_c / np.sqrt(1 - (1 - c_c) ** (calls_made - lambda_pop + 1))
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                zi = (selected[i] - x) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            C = (1 + c_c * (2 - c_c) - c_c / mu_eff) * C + c_c * rank_one + \
                (1 - c_c) * (c_sigma / d_sigma * (hs ** 2 - 1) * C - c_sigma / d_sigma * mu_eff * rank_mu)

            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive semi-definite
            eigvals_new, eigvecs_new = np.linalg.eigh(C)
            eigvals_new = np.maximum(eigvals_new, 1e-12)
            C = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
            eigvals, eigvecs = eigvals_new, eigvecs_new

            # Update center
            x = mean_y

        return best_x, best_f
