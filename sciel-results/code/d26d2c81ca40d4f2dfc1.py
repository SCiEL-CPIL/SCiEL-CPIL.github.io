"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard and budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # 30s margin before 320s kill

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * dim)  # population size
        mu = lambda_ // 2  # number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff + 2)
        c_mu = min(1 - c_1, (2 + (mu_eff / dim - 2) ** 2) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_mu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial sigma based on domain size
        sigma = 1.0
        x_mean = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = 0.0
        one_over_sqrt_detC = 1.0

        # Precompute eigendecomposition update frequency
        eig_interval = max(1, int(1 / (c_1 + c_mu) * dim / 5))
        eig_counter = 0

        # Initial evaluation
        calls = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls += 1

        # Generate initial population
        def sample_population():
            """Sample lambda points from N(x_mean, sigma^2 * C)."""
            # Eigendecomposition for sampling
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            points = x_mean[None, :] + sigma * (B @ (D[None, :] * rng.standard_normal((lambda_, dim))).T).T
            points = np.clip(points, -5.0, 5.0)
            return points

        rng = np.random.default_rng(0)

        # Main loop
        while calls < budget:
            if time.time() > deadline - 5.0:
                break

            # Sample population
            points = sample_population()

            # Evaluate
            fvals = np.empty(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    fvals[i] = best_f
                    continue
                fvals[i] = func(points[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = points[i].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            points = points[order]
            fvals = fvals[order]

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * points[:mu], axis=0)

            # Update evolution paths
            y = (x_mean - x_old) / sigma
            pc = (1 - c_c) * pc + c_c * y
            norm_pc = np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * 1))  # approximation
            ps = (1 - d_sigma) * ps + np.sqrt(d_sigma * (2 - d_sigma) * mu_eff) * y

            # Update sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - d_sigma) ** 2)) / chi_n < 1.4 + 2 / (dim + 1)
            sigma *= np.exp((1 / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Update covariance
            if hsig:
                ps = (1 - d_sigma) * ps + np.sqrt(d_sigma * (2 - d_sigma) * mu_eff) * y
            C = (1 - c_1 - c_mu) * C
            if hsig:
                C += c_1 * (2 + c_1 / c_mu) * np.outer(pc, pc)
            C += c_mu * np.sum(weights[:, None, None] * np.outer(points[:mu] - x_old, points[:mu] - x_old), axis=0) / sigma ** 2

            # Eigendecomposition for sampling (periodic)
            eig_counter += 1
            if eig_counter >= eig_interval:
                eig_counter = 0
                # Ensure positive definite
                try:
                    D, B = np.linalg.eigh(C)
                    D = np.maximum(D, 1e-12)
                    C = B @ np.diag(D) @ B.T
                except:
                    C = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
