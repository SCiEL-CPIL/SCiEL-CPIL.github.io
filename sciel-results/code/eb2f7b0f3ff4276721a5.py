"""Rewritten optimizer using CMA-ES for black-box continuous optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Time guard: return early if we're running out of time
        start_time = time.time()
        deadline = start_time + 300  # 300s limit

        # CMA-ES parameters
        # lambda: number of offspring
        lam = max(4, int(4 + 3 * d))  # for d=5, lam=19
        mu = lam // 2  # number of parents

        # Weights (standard CMA-ES weights)
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()

        # Effective population size
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))

        # Initial point: center of the box
        x0 = np.zeros(d)
        x = x0.copy()
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # Initial step size
        sigma = 2.0

        # Covariance matrix
        C = np.eye(d)
        Cinv_sqrt = np.eye(d)  # We'll use Cholesky-based update

        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)

        # Track best
        best_fx = fx

        # Precompute for efficiency
        hsig = (1 + 2 / (d + 1)) * np.sqrt((d + 1) / (d - 1) - 1)

        while calls_made < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate offspring
            Y = []
            F = []
            for i in range(lam):
                # Check time per iteration
                if time.time() > deadline - 5:
                    break
                if calls_made >= budget:
                    break

                # Sample from N(0, C) using current C
                # Use Cholesky decomposition for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(d)

                z = rng.standard_normal(d)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls_made += 1

                Y.append(y)
                F.append(fy)

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            if len(Y) == 0:
                break

            Y = np.array(Y)
            F = np.array(F)

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Update best
            if F[0] < best_fx:
                best_fx = F[0]
                best_x = Y[0].copy()

            # Mean update
            x_old = x.copy()
            x_new = np.sum(weights[:, None] * Y[:mu], axis=0)

            # Update evolution paths
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ ((x_new - x_old) / sigma))

            # hsig check
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - budget + 1) / budget)) < hsig:
                hsig = 1.0
            else:
                hsig = 0.0

            # pc update
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_new - x_old) / sigma

            # Covariance matrix update
            v = (x_new - x_old) / sigma
            rank1 = np.outer(v, v)
            rankmu = np.sum(weights[:, None, None] * (Y[:mu] - x_old)[:, :, None] * (Y[:mu] - x_old)[:, None, :], axis=0) / sigma ** 2

            C = (1 + c1 + cmu - c1 - cmu) * C + c1 * (rank1 + (1 - hsig) * cc * (1 - cc) * np.outer(pc, pc)) + cmu * rankmu

            # Ensure C is positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update Cinv_sqrt approximation
            try:
                Cinv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                Cinv_sqrt = np.eye(d)

            # Sigma update
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) - 1 / (25 * d ** 2))
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10)

            x = x_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
