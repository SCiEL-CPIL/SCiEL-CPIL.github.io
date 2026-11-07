"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialization
        rng = np.random.default_rng()
        x = np.zeros(dim)
        sigma = 2.0

        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        def get_Cinv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        Cinv_sqrt = get_Cinv_sqrt(C)

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        while calls < budget:
            x_old = x.copy()

            # Generate and evaluate population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (Cinv_sqrt @ z)
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y.copy()

            if calls >= budget:
                break

            # Sort
            order = np.argsort(F)
            Ys = Y[order]
            Fs = F[order]

            # Update mean
            x = np.sum(weights[:, None] * Ys[:mu], axis=0)

            # Compute z for step size adaptation
            z = (x - x_old) / sigma
            z_c = Cinv_sqrt @ z

            # Update path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_c

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * z

            # Update covariance
            delta_dist = (1 + (1 - c_1) * (1 - c_c)) * c_c
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                zi = Cinv_sqrt @ ((Ys[i] - x_old) / sigma)
                rankmu += weights[i] * np.outer(zi, zi)

            C = (1 + c_1 * (delta_dist - 1) - c_mu) * C + c_1 * rank1 + c_mu * rankmu

            # Ensure symmetry and positive semi-definiteness
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            Cinv_sqrt = get_Cinv_sqrt(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
