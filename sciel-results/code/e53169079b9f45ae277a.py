"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-aware budget management for BBOB f21 (101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Standard CMA-ES: population size lambda = 4 + floor(3*log(dim))
        lam = 4 + int(3 * np.log(dim))
        # Step size sigma
        sigma = 2.0
        # Initial mean: center of box
        mean = np.zeros(dim)

        # CMA-ES parameters
        cm = np.eye(dim)
        step_size = 1.0

        # Covariance matrix
        C = np.eye(dim)
        # Precompute Cholesky factor
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        # Adaptation parameters
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        iter_count = 0
        best_x = mean.copy()
        best_f = func(best_x)

        # Track evaluations used
        evals_used = 1

        while evals_used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)

            # Evaluate population
            fs = []
            for y in ys:
                if evals_used >= budget:
                    break
                f = func(y)
                fs.append(f)
                evals_used += 1
                # Check time during evaluation
                if time.time() > deadline:
                    break

            if len(fs) < 2:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # Update mean
            new_mean = np.sum(weights[:, None] * np.array(ys[:mu]), axis=0)

            # Update evolution paths
            old_mean = mean
            mean = new_mean

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (mean - old_mean) / sigma)

            # Update C
            c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma
            delta_h = 1.0 if (np.dot(p_sigma, p_sigma) / (1 - (1 - c_sigma)**(2 * (iter_count + 1)))) < (1 + 4 / dim) * chi_N**2 else 0.0

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C
            C += c_1 * (np.outer(c, c) - np.eye(dim))
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(ys[i] - old_mean, ys[i] - old_mean)

            # Ensure positive definite
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small diagonal
                C += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    C = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            iter_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
