"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of weighted parents

        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)

        # Step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) ** 2) / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, (mueff - 2 + 2 / dim) / ((dim + 2) ** 2 + mueff))

        # Initial state
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for sampling
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)

        call_count = 1
        generation = 0

        while call_count < budget:
            if time.time() > deadline:
                break

            generation += 1

            # Update eigen decomposition if needed (every few generations for speed)
            if generation % 5 == 1 or generation == 1:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                # Cholesky-like: sqrt of eigenvalues
                sqrt_eigvals = np.sqrt(eigvals)

            # Sample population
            offsets = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Transform: eigvecs @ diag(sqrt_eigvals) @ z
                y = eigvecs @ (sqrt_eigvals * z)
                offsets.append(sigma * y)

            # Evaluate population
            offspring = []
            fvalues = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                y = np.clip(x + offsets[i], -5, 5)
                fy = func(y)
                offspring.append(y)
                fvalues.append(fy)
                call_count += 1

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            if call_count < budget and len(fvalues) < lambda_:
                break

            # Sort by fitness
            indices = np.argsort(fvalues)
            sorted_offspring = [offspring[i] for i in indices[:mu]]
            sorted_fvalues = [fvalues[i] for i in indices[:mu]]

            # Weighted mean of best mu
            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += w[i] * sorted_offspring[i]

            # Update evolution paths
            # p_sigma update
            old_sigma = sigma
            # Need to compute the normalized step
            # The step in standardized space
            step = (new_x - x) / (sigma * np.sqrt(np.diag(C)) + 1e-12)
            # More accurately, we should use the inverse square root of C
            # But for simplicity and stability, use a simplified update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * step

            # Check for sigma update
            # Norm of p_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_lambda = np.sqrt(lambda_) * (1 - 1 / (4 * lambda_) - 1 / (24 * lambda_ ** 2))
            chi_mu = np.sqrt(mueff) * (1 - 1 / (4 * mueff) - 1 / (24 * mueff ** 2))

            if norm_p_sigma / chi_lambda < 1.4 + 2 / (dim + 1):
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_lambda - 1))

            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * step

            # Covariance matrix update
            # Rank-mu update
            v = np.zeros((dim, mu))
            for i in range(mu):
                v[:, i] = (sorted_offspring[i] - x) / (sigma + 1e-12)

            # Rank-1 update term
            p_c_outer = np.outer(p_c, p_c)
            # Rank-mu update term
            rank_mu = np.outer(v, v * w)

            C = (1 + c_1 * (2 - c_1) - c_1 - c_mu) * C + c_1 * (p_c_outer - C) + c_mu * rank_mu

            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive definiteness by adding small identity if needed
            # Check eigenvalues
            eig_check = np.linalg.eigvalsh(C)
            if np.min(eig_check) < 1e-12:
                C += (1e-12 - np.min(eig_check)) * np.eye(dim)

            x = new_x

        return best_x, best_fx
# EVOLVE-BLOCK-END
