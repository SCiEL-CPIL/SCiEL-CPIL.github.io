"""Optimized continuous optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with restarts for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Initial step size
        sigma0 = 2.0

        # Weights and mu
        w = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
        w /= w.sum()
        mu = int(lam / 2)
        mueff = int(np.sum(w[:mu]) ** 2 / np.sum(w ** 2))
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Damping factor for sigma update
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Try multiple restarts
        max_restarts = 3
        per_restart_budget = budget // max_restarts

        for restart in range(max_restarts):
            # Check time
            if time.time() > deadline:
                break

            # Initialize
            if restart == 0:
                # Start at origin or random
                x = np.zeros(dim)
            else:
                # Random restart
                rng = np.random.default_rng(restart * 1000 + 42)
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES state
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            sigma = sigma0

            # Precompute
            sqrt_c1 = np.sqrt(c1)
            sqrt_cmu = np.sqrt(cmu)

            # Budget for this restart
            calls_made = 1
            while calls_made < budget:
                # Check time
                if time.time() > deadline:
                    return best_x, best_f

                # Check if we've made enough calls
                if calls_made >= budget:
                    break

                # Generate population
                B = np.random.default_rng(calls_made * 7 + restart * 10000).standard_normal((lam, dim))
                # Apply transformation
                # For simplicity, use diagonal C initially, update later
                # For efficiency in 5D, we can use full C
                try:
                    # Cholesky decomposition
                    L = np.linalg.cholesky(C)
                    Y = x[None, :] + sigma * (B @ L.T)
                except np.linalg.LinAlgError:
                    # Fallback to diagonal
                    Y = x[None, :] + sigma * B

                # Evaluate
                F = np.zeros(lam)
                for i in range(lam):
                    if time.time() > deadline:
                        return best_x, best_f
                    y = np.clip(Y[i], -5, 5)
                    F[i] = func(y)
                    calls_made += 1
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = y.copy()

                # Sort
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]

                # Update mean
                x_new = np.sum(w[:, None] * Y[:mu], axis=0)

                # Update path
                y_sigma = (x_new - x) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (calls_made + 1 / lam))) / chi < 1.4 + 2 / (dim + 1)
                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_sigma

                # Update covariance
                delta_d = (2 * hsig - 1) * cc * (1 - cc)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum(w[:, None, None] * Y[:mu] * Y[:mu], axis=0) / sigma ** 2
                C = (1 + c1 * delta_d - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)

                x = x_new

                # Convergence check
                if np.linalg.norm(x_new - x) < 1e-12:
                    break

            # End of restart
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
