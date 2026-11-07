"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage optimizer: Sobol global search then CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Stage 1: Global exploration with stratified sampling
        n_global = min(80, budget)
        best_x = None
        best_f = np.inf

        # Use a mix of random and structured points
        # First, try center and some structured points
        structured = np.array([
            [0.0] * dim,
            [1.0] * dim,
            [-1.0] * dim,
            [2.0] * dim,
            [-2.0] * dim,
        ])
        for i in range(min(len(structured), n_global // 5)):
            x = structured[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Then random sampling
        n_random = n_global - min(len(structured), n_global // 5)
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Stage 2: CMA-ES refinement
        calls_used = n_global
        remaining = budget - calls_used
        if remaining <= 0:
            return best_x, best_f

        n = dim
        # CMA-ES parameters
        lambda_ = max(4, 4 * n)  # population size
        mu = max(1, lambda_ // 2)  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_c = (1 + 3 / (n + 1)) / ((n + 1) + 3)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        for gen in range(remaining):
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lambda_, n))
            fvals = np.zeros(lambda_)
            for j in range(lambda_):
                if time.time() > deadline:
                    break
                # Sample from N(mean, sigma^2 * C)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C is not positive definite
                    L = np.eye(n)
                z = rng.standard_normal(n)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fvals[j] = func(y)
                ys[j] = y
                calls_used += 1

                if fvals[j] < best_f:
                    best_f = fvals[j]
                    best_x = y.copy()

            # Sort by fitness
            order = np.argsort(fvals)
            ys = ys[order]
            fvals = fvals[order]

            # Update mean
            new_mean = np.sum(weights[:, None] * ys[:mu], axis=0)
            delta_mean = new_mean - mean

            # Update evolution paths
            if sigma > 1e-12:
                inv_sigma = 1.0 / sigma
            else:
                inv_sigma = 1.0
            pc = (1 - c_c) * pc + c_c * inv_sigma * delta_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sigma * (C @ delta_mean)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update covariance
            delta_sigma = np.linalg.norm(ps) / chi_n - 1
            C = (1 + c_1 * delta_sigma - c_mu) * C
            for j in range(mu):
                C += c_mu * weights[j] * np.outer(ys[j] - mean, ys[j] - mean)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(n)

            mean = np.clip(new_mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
