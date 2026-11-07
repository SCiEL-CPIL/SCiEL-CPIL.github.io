"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive sigma and time guard for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin within 300s limit
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Standard CMA-ES with adaptive step size
        # For dim=5, we use population size lambda = 4 + floor(3*log(5+1)) = 4+3*1 = 7
        # But for small dim, smaller populations can work well
        lam = 7  # population size
        mu = lam // 2  # number of selected parents

        # Initial point: try center first, then random
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Initial sigma based on problem scale
        sigma = 1.0

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Inverse square root of C for step generation
        C_inv_sqrt = np.eye(dim)
        # Evolution path for covariance adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Parameters
        c_sigma = (mu + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, (mu - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            # Use Cholesky-like approach: step = sigma * C_inv_sqrt @ z
            # For efficiency, we approximate with random normal and scaling
            population = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Transform using C_inv_sqrt (approximated)
                step = sigma * (C_inv_sqrt @ z)
                x_candidate = x + step
                # Clip to box
                x_candidate = np.clip(x_candidate, -5, 5)
                population.append((x_candidate, None))

            # Evaluate population
            for i, (xc, _) in enumerate(population):
                if time.time() > deadline:
                    break
                fc = func(xc)
                population[i] = (xc, fc)
                calls += 1
                if fc < best_fx:
                    best_fx = fc
                    best_x = xc.copy()

            # Sort by fitness
            population.sort(key=lambda t: t[1])
            # Take top mu
            selected = [pop[i][0] for i in range(mu)]
            selected_f = [pop[i][1] for i in range(mu)]

            # New mean
            x_new = np.sum([weights[i] * selected[i] for i in range(mu)], axis=0)

            # Update evolution paths
            # p_sigma
            z_mean = (x_new - x) / sigma if sigma > 1e-15 else np.zeros(dim)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean

            # p_c
            x_diff = (x_new - x) / sigma if sigma > 1e-15 else np.zeros(dim)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * x_diff

            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Covariance update
            v = p_c / np.sqrt(1 - (1 - c_c) ** 2) if np.sqrt(1 - (1 - c_c) ** 2) > 1e-10 else p_c
            delta_h = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + c_sigma) * chi_n) else 0
            C = (1 + c_1 * delta_h * (mu_eff - 1) - c_1) * C + c_1 * np.outer(v, v) + \
                c_mu * np.sum([weights[i] * np.outer(selected[i] - x, selected[i] - x) for i in range(mu)], axis=0)

            # Renormalize C to maintain determinant 1 (approximate)
            # For simplicity, we'll do a simple rescaling
            det_C = np.linalg.det(C)
            if det_C > 0:
                C /= det_C ** (1.0 / dim)

            # Update C_inv_sqrt (approximate via eigendecomposition)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-15, None)
                C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(dim)

            x = x_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
