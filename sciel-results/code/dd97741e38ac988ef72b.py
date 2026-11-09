"""Rewritten optimizer: CMA-ES with adaptive restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step-size, boundary handling, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize at origin (common for BBOB-style with hidden shift in [-4,4])
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_used = 1

        # CMA-ES parameters
        # Initial sigma based on box size
        sigma = 1.0
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        # Weights
        mu = lambda_ // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        # Step-size adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        # Mean adaptation
        c_c = 4 / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Path lengths
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (initialized as identity)
        C_inv_sqrt = np.eye(dim)
        # Update counter
        count_evals = 0

        def compute_C_inv_sqrt(C):
            """Compute inverse square root of C using eigendecomposition."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        def sample_from_C(C_inv_sqrt, rng):
            """Sample z from N(0,1) and transform to N(0,C)."""
            z = rng.standard_normal(dim)
            return C_inv_sqrt @ z

        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            pop = []
            pop_f = []
            for i in range(lambda_):
                z = sample_from_C(C_inv_sqrt, rng)
                y = x + sigma * z
                # Clip to box
                y_clipped = np.clip(y, -5, 5)
                fy = func(y_clipped)
                pop.append(y_clipped)
                pop_f.append(fy)
                calls_used += 1
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

            if len(pop) < lambda_:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pop = [pop[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()

            # Mean adaptation
            x_old = x.copy()
            for i in range(mu):
                x = x + weights[i] * (pop[i] - x_old)

            # Compute evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x - x_old) / sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x - x_old) / sigma

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = 1.0 / (sigma ** 2) * np.outer(x - x_old, x - x_old)
            C = (1 + c_c * delta_h - c_c - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(pop[i] - x_old, pop[i] - x_old) for i in range(mu))

            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                C = eigvecs @ np.diag(np.clip(eigvals, 1e-12, None)) @ eigvecs.T

            # Update inverse square root periodically
            if calls_used % (lambda_ * 5) == 0:
                C_inv_sqrt = compute_C_inv_sqrt(C)

            # Step-size adaptation
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Check for convergence
            if sigma < 1e-8 or np.linalg.norm(x - x_old) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
