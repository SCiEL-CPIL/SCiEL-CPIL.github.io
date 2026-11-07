"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # CMA-ES parameters
        # For Bent Cigar (f12), the landscape is highly anisotropic.
        # CMA-ES adapts covariance to this.
        # lambda = 4 + floor(3 * ln(dim))
        lam = int(4 + 3 * np.log(dim))
        # mu = floor(lam / 2)
        mu = lam // 2
        # weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # effective population size
        mueff = 1.0 / np.sum(weights ** 2)
        # step size update parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # chi_n = expected norm of standard normal
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initial point: center of box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Initial step size
        sigma = 2.0

        # Covariance matrix (identity)
        C = np.eye(dim)
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Inverse square root of C
        C_sqrt_inv = np.eye(dim)

        # Precompute eigen decomposition update
        def update_C_sqrt_inv(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt_inv

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lam, dim))
            fs = np.zeros(lam)

            for i in range(lam):
                # Sample from N(x, sigma^2 * C)
                z = np.random.randn(dim)
                y = x + sigma * C_sqrt_inv @ z
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # Mean
            y_mean = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)

            # Update evolution paths
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ (y_mean - x) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * (y_mean - x) / sigma

            # Update covariance matrix
            delta_h = (1 + 0.3 * (mueff - dim) / (dim + mueff)) if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (budget - calls))) < (1 + 0.3 * (mueff / dim)) * chi_n else 0

            rank_mu = np.outer(y_mean - x, y_mean - x)
            for i in range(mu):
                diff = ys[i] - x
                rank_mu += weights[i] * np.outer(diff, diff)

            C = (1 + c1 * delta_h * np.outer(p_sigma, p_sigma) / dim + cmu * rank_mu / dim) * C

            # Symmetrize
            C = (C + C.T) / 2

            # Update step size
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 100)

            # Update mean
            x = y_mean

            # Update inverse square root
            C_sqrt_inv = update_C_sqrt_inv(C)

            # Check convergence
            if sigma < 1e-15:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
