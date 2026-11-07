"""CMA-ES based optimizer for f19 with time-aware budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Griewank-Rosenbrock function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial population: mix of random and structured points
        n_init = 10
        X_init = np.random.uniform(-5, 5, (n_init, dim))
        # Add some structured points (Halton-like or grid)
        for i in range(dim):
            X_init[i, i] = 0.0
        Y_init = np.array([func(x) for x in X_init])
        calls = n_init

        best_x = X_init[np.argmin(Y_init)]
        best_y = Y_init.min()

        # CMA-ES parameters
        pop_size = max(4, dim + int(3 * np.log(dim)))
        sigma = 2.0  # Initial step size
        x_mean = best_x.copy()

        # CMA-ES state
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Precompute
        mu = pop_size // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Mahalanobis transform using Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                x_new = x_mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                y_new = func(x_new)
                calls += 1
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()

                # Store for CMA-ES update
                if not hasattr(self, '_cma_cache'):
                    self._cma_cache = []
                self._cma_cache.append((x_new, y_new, z))

            if len(self._cma_cache) < pop_size:
                break

            # Sort by fitness
            self._cma_cache.sort(key=lambda t: t[1])
            selected = self._cma_cache[:mu]
            self._cma_cache = []

            # Compute new mean
            x_mean = np.sum([w * x for w, (x, y, z) in zip(weights, selected)], axis=0)

            # Compute covariance update
            y_vec = np.zeros(dim)
            for w, (x, y, z) in zip(weights, selected):
                y_vec += w * (x - x_mean) / sigma

            # Update step-size control
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) / mu_eff) * (np.linalg.norm(y_vec) / np.sqrt(mu_eff) - 1) * np.sign(np.linalg.norm(y_vec) > 1e-10)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Update evolution path
            c_inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(dim)))
            p_c = (1 - c_c) * p_c + c_c * c_inv_sqrt_C @ y_vec
            chi_N = np.sqrt(dim) * (1 - 1 / (2 * dim) + 1 / (12 * dim ** 2))

            # Update covariance
            delta = (1 + 2 * max(0, (np.linalg.norm(p_c) ** 2 / chi_N ** 2 - 1) * c_c / (1 - c_c))) * c_mu
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) - C) + c_mu * np.sum([w * np.outer(x - x_mean, x - x_mean) / sigma ** 2 for w, (x, y, z) in zip(weights, selected)], axis=0)

            # Symmetrize
            C = (C + C.T) / 2

            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C = C + 1e-10 * np.eye(dim)

        return best_x, best_y
# EVOLVE-BLOCK-END
