"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style continuous optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                return np.inf
            calls_used += 1
            try:
                return float(func(np.asarray(x, dtype=np.float64)))
            except Exception:
                return np.inf

        # Initial diverse starts
        n_starts = min(10, self.budget // 20)
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_call(best_x)

        # CMA-ES-like optimization
        dim = self.dim
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Initial sigma
        sigma = 1.0
        x_mean = best_x.copy()
        f_best = best_f

        # Covariance matrix
        C = np.eye(dim)
        # Step size adaptation
        cs = 1 / (lam + 3)
        ds = 1 + 2 * max(0, (dim + 1) / (lam + 7) - 1)
        hsig = (1 + 2 / (dim + 1)) * np.sqrt((lam + 1) / (lam - 1))

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f

        for gen in range(1, self.budget // lam + 1):
            if time.time() > deadline - 2.0:
                break
            if calls_used >= self.budget:
                break

            # Generate offspring
            offspring = []
            offspring_f = []
            for i in range(lam):
                if calls_used >= self.budget:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Transform using Cholesky of C
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                offspring.append(y)
                offspring_f.append(fy)

            if len(offspring) < 2:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Update best
            if sorted_f[0] < global_best_f:
                global_best_f = sorted_f[0]
                global_best_x = sorted_offspring[0].copy()

            # Weighted recombination
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * sorted_offspring[i]

            # Update evolution paths
            diff = (x_new - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * diff
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mueff) * diff

            # Update covariance
            c1 = (mueff + 2) / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            outer_products = np.zeros((dim, dim, mu))
            for i in range(mu):
                outer_products[:, :, i] = np.outer(sorted_offspring[i] - x_mean,
                                                   sorted_offspring[i] - x_mean)
            C = (1 + c1 * (2 - cs - cs / (2 * dim)) + cmu - c1 - cmu) * C \
                + c1 * np.outer(p_c, p_c) \
                + cmu / (sigma ** 2) * np.sum(weights[:, None, None] * outer_products, axis=2)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x_mean = x_new

            # Regularize C periodically
            if gen % 10 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                C = (C + C.T) / 2

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
