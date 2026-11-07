"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and rotation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize center
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Adaptive step size (sigma)
        sigma = 3.0

        # Covariance matrix (start with identity)
        C = np.eye(n)
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)

        # Path accumulators for covariance adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Update rates
        c_sigma = 1.0 / (n + 3)
        c_c = (4 + n / 2) / (n + 4)
        c_1 = 2 / ((n + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * ((n + 1.5) / (n + 2)) ** 2 - 1)

        # Population size
        lambda_pop = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_pop // 2

        # Weighting
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Counter for covariance update
        count = 0

        for i in range(budget - 1):
            # Sample population
            Y = np.zeros((lambda_pop, n))
            for k in range(lambda_pop):
                # Sample from N(0, C)
                z = rng.standard_normal(n)
                # Transform: eigvecs @ sqrt(eigvals) @ z
                Y[k] = x + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                Y[k] = np.clip(Y[k], -5, 5)

            # Evaluate
            F = np.array([func(y) for y in Y])

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()

            # Weighted mean of best mu solutions
            y_w = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update step size (1/5 rule variant with path)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                (eigvecs.T @ (eigvecs @ y_w)) / sigma
            )
            # Note: p_sigma is in transformed space, but let's keep it simple in original space for now
            # Actually, let's just use a simpler adaptive sigma based on success rate

            # Simpler sigma adaptation: 1/5 rule
            success = F_sorted[0] < fx
            if success:
                sigma *= 1.2
            else:
                sigma *= 0.85
            sigma = np.clip(sigma, 1e-4, 10.0)

            # Update center
            x = y_w
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Update covariance (simplified: only if we have enough data)
            count += 1
            if count % 10 == 0:
                # Update C using rank-1 update
                v = y_w / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(v, v) + (1 - (1 - c_sigma) ** 2) * np.eye(n)) + c_mu * np.sum(
                    weights[:, None, None] * np.outer(Y_sorted[:mu] - x, Y_sorted[:mu] - x) / sigma ** 2, axis=0
                )

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                # Normalize to maintain determinant
                det = np.prod(eigvals) ** (1.0 / n)
                eigvals /= det

        return best_x, best_f
# EVOLVE-BLOCK-END
