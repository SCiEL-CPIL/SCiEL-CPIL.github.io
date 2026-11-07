"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size
        lambda_pop = max(4 + 3 * int(np.log(dim)), 10)
        # Number of parents
        mu = lambda_pop // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effective selection mass
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        # Step size control
        cs = (mueff + 1.5) / (dim + mueff + 5)
        # Covariance matrix adaptation
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (1 - ccs)
        cmu = (2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)) * (
            min(1 - c1, 2 / ((dim + 1.3) ** 2 + mueff))
        )
        ccs = (mueff + 1) / (dim + mueff + 2)

        # Initial state
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        Cinv = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute Cholesky factor
        B, D = self._cholesky(C)

        best_x = x
        best_f = func(x)
        calls = 1

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            z = rng.standard_normal((lambda_pop, dim))
            y = x + sigma * np.dot(z, B.T) * D
            # Clip to box
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fy = np.array([func(yi) for yi in y])
            calls += lambda_pop

            # Sort by fitness
            order = np.argsort(fy)
            y = y[order]
            fy = fy[order]

            # Update best
            if fy[0] < best_f:
                best_f = fy[0]
                best_x = y[0].copy()

            # Update mean
            x_new = np.dot(weights, y[:mu])
            
            # Update step size
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (
                np.dot(B.T, D * (np.dot(x_new - x, B) / sigma))
            )
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls - lambda_pop))) < (1.4 + 2 / (dim + 1)) * cs:
                sigma *= np.exp(cs * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))

            # Update covariance
            p_c = (1 - ccs) * p_c + np.sqrt(ccs * (2 - ccs) * mueff) * (x_new - x) / sigma
            
            delta = 0
            if np.linalg.norm(p_c) / np.sqrt(1 - (1 - ccs) ** calls) < (1.4 + 2 / (dim + 1)) * ccs:
                delta = 1
                
            C = (1 + c1 * delta - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * np.sum(
                weights[:, np.newaxis, np.newaxis] * np.outer(y[:mu] - x, y[:mu] - x), axis=0
            ) / sigma**2
            
            # Recompute Cholesky
            B, D = self._cholesky(C)
            x = x_new

        return best_x, best_f

    def _cholesky(self, C):
        try:
            L = np.linalg.cholesky(C)
            D = np.diag(L)
            B = L / D[:, np.newaxis]
            return B, D
        except np.linalg.LinAlgError:
            # Fallback if matrix becomes non-positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(C)
            D = np.diag(L)
            B = L / D[:, np.newaxis]
            return B, D
# EVOLVE-BLOCK-END
