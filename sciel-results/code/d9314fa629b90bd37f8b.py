"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for Rastrigin f3 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Initial population size
        lam = max(4, int(4 + 3 * np.log(n)))  # ~14 for n=5
        # Cap lambda so we don't exceed budget
        if lam > budget // 2:
            lam = max(4, budget // 2)

        # Initialize center
        c = np.zeros(n)
        fc = func(c)

        # Step size
        sigma = 2.0

        # Covariance matrix
        C = np.eye(n)

        # Evolution path for sigma
        cs = (n + 2) / (n + 8)
        c_sigma = 2 * (n + 0.5) / (n + 12)
        ps = np.zeros(n)

        # Evolution path for covariance
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 2)
        cmu = min(0.3 - c1, 2 * (2 - (n + 2) / (n + 8)) / ((n + 2) ** 2 + 2))
        pc = np.zeros(n)

        # Precompute constants
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        best_x = c.copy()
        best_f = fc
        calls_made = 1

        # We need to track all evaluations
        # CMA-ES: each generation evaluates lam points
        # Total evaluations = 1 (initial) + generations * lam
        # We have budget-1 remaining evaluations

        max_gens = (budget - 1) // lam
        if max_gens < 1:
            # Fallback to simple random search if budget too small
            rng = np.random.default_rng(42)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        rng = np.random.default_rng(12345)

        for gen in range(max_gens):
            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(n)
                # Transform using Cholesky of C
                # For efficiency, we can use eigen decomposition or just sqrtm
                # Since n=5, we can do Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                y = c + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            fvals = []
            for y in offspring:
                fy = func(y)
                fvals.append(fy)
                calls_made += 1

            fvals = np.array(fvals)

            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = offspring[idx_min].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            mu = lam // 2  # number of parents
            if mu < 1:
                mu = 1

            # Compute weights
            weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights ** 2)

            # Update mean
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * offspring[order[i]]
            c_new = c + y_w
            c_new = np.clip(c_new, -5, 5)

            # Update evolution path for sigma
            # Compute C^{-1/2} (y_w)
            try:
                L_inv = np.linalg.inv(np.linalg.cholesky(C))
                y_w_trans = L_inv.T @ y_w
            except np.linalg.LinAlgError:
                y_w_trans = y_w

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_w_trans

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution path for covariance
            pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mueff) * y_w_trans

            # Update covariance
            delta_h = (1.4 + 2 / (n + 1)) * (1 - c1)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                z_i = np.zeros(n)
                try:
                    z_i = L_inv.T @ (offspring[order[i]] - c) / sigma
                except np.linalg.LinAlgError:
                    z_i = (offspring[order[i]] - c) / sigma
                rank_mu += weights[i] * np.outer(z_i, z_i)

            C = (1 - c1 - cmu) * C + c1 * (rank_one + (1 - delta_h) * np.eye(n)) + cmu * rank_mu

            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)

            c = c_new

        return best_x, best_f
# EVOLVE-BLOCK-END
