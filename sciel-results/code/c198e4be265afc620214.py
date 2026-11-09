"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # Population size
        mu = lambda_ // 2  # Number of weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = (2.0 / ((dim + 1.3) ** 2 + mueff)) / (dim + 6.0)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))

        # Initial state
        x = np.zeros(dim)
        f_best = func(x)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        inv_sqrt_C = np.eye(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))

        calls_made = 1
        best_x = x.copy()
        best_f = f_best

        while calls_made < budget:
            # Time check
            if time.time() > deadline:
                break

            # Generate population
            Y = []
            for _ in range(lambda_):
                z = np.random.randn(dim)
                y = inv_sqrt_C @ z
                Y.append(x + sigma * y)

            # Evaluate population
            F = []
            for y in Y:
                y_clipped = np.clip(y, -5, 5)
                f_val = func(y_clipped)
                F.append(f_val)
                calls_made += 1
                if calls_made >= budget:
                    break

            # Sort by fitness
            idx = np.argsort(F)
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]

            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()

            # Update mean
            x_new = x.copy()
            for i in range(mu):
                x_new += w[i] * Y[i]
            x = x_new

            # Update covariance matrix
            yvec = (x_new - x) / sigma
            hsig = (1 - np.prod(1 - cc) ** (1 / cs)) * (np.linalg.norm(ps) / chi_n - (1 - cc)) < 2 + 2 / (dim + 1)
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * yvec
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yvec

            rank_mu = np.outer(yvec, yvec)
            for i in range(1, mu):
                yi = (Y[i] - x) / sigma
                rank_mu += w[i] * np.outer(yi, yi)

            C = (1 + c1 * (hsig - 1) + c1 * mueff) * C - c1 * np.outer(pc, pc) + cmu * rank_mu
            C = 0.5 * (C + C.T)  # Symmetrize

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Update inverse square root of C
            D, V = np.linalg.eigh(C)
            D = np.clip(D, 1e-12, None)
            inv_sqrt_C = V @ np.diag(1.0 / np.sqrt(D)) @ V.T

        return best_x, best_f
# EVOLVE-BLOCK-END
