"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Strategy parameters
        # For n=5, mu=2 is a good balance
        mu = max(1, n // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / (n + 2)) / ((n + 2) ** 2 + mu_eff))

        lb = -5.0
        ub = 5.0

        # Initialize at a random point to avoid local optima at origin
        x_mean = rng.uniform(lb, ub, size=n)
        sigma = 2.0
        
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)

        calls = 0
        best_x = x_mean.copy()
        best_f = float('inf')

        # Evaluate initial point
        fx0 = func(x_mean)
        calls += 1
        best_f = fx0
        best_x = x_mean.copy()

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(n)

        while calls < budget:
            # Generate mu offspring
            offspring = []
            for i in range(mu):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, lb, ub)
                offspring.append(y)

            # Evaluate all offspring
            offspring_f = []
            for y in offspring:
                if calls >= budget:
                    break
                fy = func(y)
                calls += 1
                offspring_f.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if calls >= budget:
                break

            if len(offspring_f) < mu:
                # If we ran out of budget mid-generation, pad with best found so far
                # This is a fallback, shouldn't happen if we check budget before loop
                while len(offspring_f) < mu:
                    offspring_f.append((best_f, best_x.copy()))

            # Sort by fitness (ascending)
            offspring_f.sort(key=lambda x: x[0])
            
            # Select the best mu individuals
            # Note: standard CMA-ES uses the best mu out of lambda. Here lambda=mu.
            # So we just use all of them.
            
            # Update mean
            # Vectorized weighted sum
            x_new = np.dot(w, np.array([y for _, y in offspring_f]))

            # Update step size control variable
            # C^{-1/2} approximation using Cholesky
            try:
                # L is such that L @ L.T = C. So C^{-1/2} is not simply inv(L).
                # We need C^{-1/2} @ v.
                # If C = L L^T, then C^{-1} = L^{-T} L^{-1}.
                # C^{-1/2} is the matrix square root of C^{-1}.
                # For small n, we can use eigen decomposition or just use the Cholesky based update if we reformulate.
                # Standard CMA-ES uses C^{-1/2} for the path update.
                # Let's compute C_inv_sqrt properly.
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)

            # Update evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new - x_mean) / sigma)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update evolution path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma

            # Update covariance matrix
            # C = (1 + c1 + cmu - c1)C + c1(pc pc^T - C) + cmu sum w_i (y_i - x_mean)(y_i - x_mean)^T / sigma^2
            # Simplifies to: C = (1 + c1 + cmu)C - c1 C + c1 pc pc^T + cmu * rank_mu_term
            # Wait, the standard formula is:
            # C = (1 + c1 + cmu - c1)C + c1(pc pc^T - C) + cmu * H
            # where H = sum w_i (y_i - x_mean)(y_i - x_mean)^T / sigma^2
            # So C_new = (1 + cmu) C + c1 (pc pc^T - C) + cmu H
            # C_new = (1 + cmu) C - c1 C + c1 pc pc^T + cmu H
            # C_new = (1 + cmu - c1) C + c1 pc pc^T + cmu H
            
            H = np.zeros((n, n))
            for i in range(mu):
                y_i = offspring_f[i][1]
                diff = (y_i - x_mean) / sigma
                H += w[i] * np.outer(diff, diff)
            
            C = (1 + c_mu - c_1) * C + c_1 * np.outer(pc, pc) + c_mu * H

            # Ensure C is positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)

            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
