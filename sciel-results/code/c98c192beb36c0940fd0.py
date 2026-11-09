"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style evolution strategy with adaptive step size and population."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)

        # Adaptive parameters
        sigma = 1.0
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = max(2, 4 + int(3 * np.log(dim)))
        # Number of selected individuals: mu = lam // 2
        mu = max(1, lam // 2)
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance effective number of parents
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size control parameters
        cs = (mueff + 1.5) / (dim + mueff + 3)
        ds = 1 + 2 * max(0, (mueff - 2) / (dim + mueff))
        # Covariance matrix adaptation
        c1 = (2 / ((dim + 1.3) ** 2 + mueff))
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Correlation threshold
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        # Precompute inverse square root of C for generating steps
        # We'll use a simple approach: generate steps in the standard normal space
        # and transform them using the Cholesky factor of C.

        def inv_sqrt(C):
            # Compute inverse square root of symmetric positive definite matrix C
            # Using eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return inv_sqrt

        C_sqrt_inv = inv_sqrt(C)

        # Track best solution
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        for i in range(budget - 1):
            if calls_made >= budget:
                break

            # Update step size control
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ (x - x)  # placeholder
            # Actually, we need the mean of the selected points
            # Let's restructure: generate population, evaluate, select, update

            # Generate population
            offspring = []
            for j in range(lam):
                # Generate step
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt_inv @ z
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                fy = func(y)
                calls_made += 1
                offspring_f.append(fy)

            offspring_f = np.array(offspring_f)
            offspring = np.array(offspring)

            # Select best mu
            idx = np.argsort(offspring_f)[:mu]
            y_selected = offspring[idx]
            f_selected = offspring_f[idx]

            # Update mean
            x_new = np.sum(weights[:, None] * y_selected, axis=0)

            # Update state variables
            # p_sigma update
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ (x_new - x)

            # p_c update
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mueff) * (x_new - x) / sigma

            # Update C
            # Rank-one update
            v = p_c / np.linalg.norm(p_c) if np.linalg.norm(p_c) > 1e-10 else np.zeros(dim)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (i + 1)) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 2) else 0.0
            C = (1 + c1 * h_sigma * (h_sigma - 1) - c1) * C + c1 * np.outer(v, v) + cmu * np.sum(weights[:, None, None] * np.outer(y_selected - x, y_selected - x) / sigma ** 2, axis=0)

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update inverse square root
            C_sqrt_inv = inv_sqrt(C)

            # Step size update
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp((cs / ds) * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Accept new mean
            x = x_new
            # Note: we don't have fx for x_new directly, but we can track the best from offspring

            # Update best
            min_idx = np.argmin(offspring_f)
            if offspring_f[min_idx] < best_f:
                best_f = offspring_f[min_idx]
                best_x = offspring[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
