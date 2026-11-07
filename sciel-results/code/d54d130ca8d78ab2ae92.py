"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES with adaptive population and deadline guard.
    Designed for BBOB f19 (Griewank-Rosenbrock) on [-5,5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Best solution tracking
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        calls_made = 1

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * ln(dim))
        lam = int(4 + 3 * np.log(dim))
        sigma = 2.5  # initial step size

        # Strategy parameters
        alpha_mu = 0.5  # learning rate for sigma
        c_sigma = (mu_eff + 2) / ((dim + 2) * mu_eff + 4)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu = lam // 2  # number of selected individuals
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

        # Recalculate c_sigma and c_c with correct mu
        c_sigma = (mu_eff + 2) / ((dim + 2) * mu_eff + 4)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)

        # Covariance matrix adaptation
        c_cov_pos = (4 + mu_eff / dim) / (dim ** 2 + 8)
        c_cov_neg = 2 * (2 - mu_eff / dim) / (dim ** 2 + 13)
        d_cov = dim + 2 * np.sqrt((mu_eff / dim - 1) * (1 - c_cov_neg / c_cov_pos)) - 4

        # State variables
        x_mean = x_best.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

        # Restart logic: track stagnation
        stagnation_counter = 0
        f_best_prev = f_best
        restart_count = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((lam, dim))
            # Whitening transformation
            # Use eigenvalue decomposition for C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            Y = B @ C_sqrt_inv
            X = x_mean + sigma * Y
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.array([func(x) for x in X])
            calls_made += lam

            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]

            # Update best
            if F_sorted[0] < f_best:
                f_best = F_sorted[0]
                x_best = X_sorted[0].copy()

            # Selection
            X_sel = X_sorted[:mu]

            # Update mean
            x_mean_new = np.sum(weights[:, None] * X_sel, axis=0)

            # Update path accumulators
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_sqrt_inv.T @ (x_mean_new - x_mean) / sigma
            cs = np.linalg.norm(pc) / chi_n
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_mean_new - x_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lam)))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Update sigma
            sigma *= np.exp(alpha_mu / c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update covariance matrix
            delta_sig = (1 - hsig) * c_sigma * (2 - c_sigma)
            rank_one = np.outer(ps, ps)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = C_sqrt_inv.T @ (X_sel[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)

            C = (1 + c_cov_pos * c_c * (1 - c_c) + c_cov_neg - c_cov_pos) * C \
                + c_cov_pos * (c_c * (2 - c_c) * np.outer(pc, pc) + delta_sig * rank_one - rank_mu)

            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_mean = x_mean_new

            # Stagnation detection
            if f_best < f_best_prev - 1e-6:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            f_best_prev = f_best

            # Restart if stuck for too long
            if stagnation_counter > 100 and calls_made < budget - 50:
                restart_count += 1
                # Reset CMA-ES state, keep best found so far
                x_mean = x_best.copy()
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                sigma = 2.5 / (restart_count + 1)
                stagnation_counter = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
