"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with covariance adaptation for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        # CMA-ES parameters
        lam = max(2 * dim, 9)  # population size
        mu = lam // 2          # number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        rng = np.random.default_rng(0)
        x = rng.uniform(-2, 2, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        while calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            Y = rng.standard_normal((lam, dim))
            X = np.array([x + sigma * (Y[i] @ B.T) for i in range(lam)])
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(X[i]) for i in range(lam)])
            calls += lam

            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]

            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()

            # Weighted mean of best mu
            m = np.sum(weights[:, None] * X[:mu], axis=0)

            # Update evolution paths
            pc = (1 - cc) * pc + cc / np.sqrt(cc) * (np.linalg.inv(B) @ (m - x) / sigma)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(pc) * (m - x) / sigma

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            damps = 1 + 2 * max(0, (mueff - 1) / (dim + 1) - 1) + cs

            # Update covariance
            delta_sigma = (1 + 2 * (cs / damps) * (np.linalg.norm(ps) / chi_n - 1)) - 1
            C = (1 + c1 + cmu - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) - C) \
                + cmu / sigma**2 * np.sum(weights[:, None, None] * np.outer(X[:mu] - m, X[:mu] - m), axis=0)

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = (eigvecs * eigvals) @ eigvecs.T

            x = m

        return best_x, best_f
# EVOLVE-BLOCK-END
