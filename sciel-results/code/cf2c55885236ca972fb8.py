"""CMA-ES with adaptive population size for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initialize at center with some spread
        x_mean = np.zeros(dim)
        f_mean = func(x_mean)
        best_x = x_mean.copy()
        best_f = f_mean
        calls_used = 1

        # CMA-ES parameters
        # Start with larger population, adaptively shrink
        lam = max(2 * dim, 6)  # Initial population size
        sigma = 2.0
        C = np.eye(dim)
        C_sqroot = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        mu = lam // 2  # Number of weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[1] + 2 * sum(weights[2:mu - 1]) - 1) / ((dim + 2) ** 2 + mu_eff))

        # Hessian of log det
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Adaptive population: shrink as we converge
        base_lam = lam
        min_lam = max(4, dim)

        for gen in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Adapt population size based on progress
            # Start large, shrink as sigma decreases or progress stalls
            progress_factor = min(1.0, sigma / 2.0)
            current_lam = max(min_lam, int(base_lam * (0.5 + 0.5 * progress_factor)))
            current_mu = max(2, current_lam // 2)

            # Update weights for current population
            w = np.log(current_mu + 1) - np.log(np.arange(1, current_mu + 1))
            w /= w.sum()
            mu_eff_cur = 1.0 / np.sum(w ** 2)

            # Update adaptation parameters for current population
            c_sigma_cur = (mu_eff_cur + 2) / (dim + mu_eff_cur + 5)
            d_sigma_cur = 1 + 2 * max(0, (mu_eff_cur - 1) / (dim + 1)) + c_sigma_cur
            c_c_cur = (4 + mu_eff_cur / dim) / (dim + 4 + 2 * mu_eff_cur / dim)
            c_1_cur = 2 / ((dim + 1.3) ** 2 + mu_eff_cur)
            c_mu_cur = min(1 - c_1_cur, 2 * (w[1] + 2 * sum(w[2:current_mu - 1]) - 1) / ((dim + 2) ** 2 + mu_eff_cur)) if current_mu > 2 else 1 - c_1_cur

            # Generate population
            offspring = []
            for i in range(current_lam):
                if calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                y = np.clip(x_mean + sigma * (C_sqroot @ z), -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append((fy, y))

            if len(offspring) == 0:
                break

            # Sort by fitness
            offspring.sort(key=lambda x: x[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Update mean (using top mu)
            y_w = np.zeros(dim)
            for i in range(min(current_mu, len(offspring))):
                y_w += w[i] * offspring[i][1]
            x_new = np.clip(y_w, -5, 5)

            # Update evolution paths
            y_sigma = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma_cur) * p_sigma + np.sqrt(c_sigma_cur * (2 - c_sigma_cur) * mu_eff_cur) * y_sigma
            p_c = (1 - c_c_cur) * p_c + np.sqrt(c_c_cur * (2 - c_c_cur) * mu_eff_cur) * y_sigma

            # Update sigma
            sigma *= np.exp(c_sigma_cur / d_sigma_cur * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance matrix
            delta_sig = (1 + 2 * max(0, (np.linalg.norm(p_sigma) / chi_n - 1) / (c_sigma_cur * d_sigma_cur))) - 1
            rank_one = np.outer(y_sigma, y_sigma)
            C = (1 + c_1_cur * delta_sig - c_1_cur - c_mu_cur) * C
            C += c_1_cur * (rank_one + delta_sig * np.eye(dim))

            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(min(current_mu, len(offspring))):
                y_i = (offspring[i][1] - x_mean) / sigma
                rank_mu += w[i] * np.outer(y_i, y_i)
            C += c_mu_cur * rank_mu

            # Ensure C is positive definite and update square root
            C = (C + C.T) / 2
            # Eigenvalue clipping for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, 1e6)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            # Compute square root via Cholesky with fallback
            try:
                C_sqroot = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: use eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, 1e6)
                C_sqroot = eigvecs @ np.diag(np.sqrt(eigvals))

            x_mean = x_new

            # Early termination if converged
            if sigma < 1e-8 or np.all(np.abs(x_mean - best_x) < 1e-6):
                # Do a few more local refinements
                for _ in range(min(10, budget - calls_used)):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(best_x + 0.1 * sigma * z, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
