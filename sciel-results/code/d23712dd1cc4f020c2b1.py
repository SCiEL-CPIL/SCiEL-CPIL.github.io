"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


class Optimizer:
    """Multi-restart CMA-ES with time guard for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Global best tracking
        global_best_x = None
        global_best_f = float('inf')

        # Use a portion of budget for initial sampling to get a good starting point
        # Then use CMA-ES with restarts

        # Phase 1: Initial random sampling (10% of budget)
        n_init = max(10, budget // 10)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)

        # Phase 2: CMA-ES with restarts
        remaining_budget = budget - n_init
        # Split remaining budget among restarts
        n_restarts = max(1, min(5, remaining_budget // 50))
        budget_per_restart = remaining_budget // n_restarts

        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if budget_per_restart < 10:
                break

            # Initialize CMA-ES parameters
            # For restart 0, start from global best; for others, random
            if restart == 0:
                x_mean = global_best_x.copy()
            else:
                x_mean = rng.uniform(-5, 5, dim)
                fx = func(x_mean)
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x_mean.copy()

            sigma = 1.0
            # Population size
            lam = max(4, int(4 + 3 * np.log(dim)))
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Adaptation parameters
            c_sigma = (lam / 2 + 1) / (dim + 2 + lam / 2)
            mu_sigma = (lam - 2) / (lam / 2 + 1)
            c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
            mu_c = (lam - 2 + 0.5 * lam / dim) / (dim + 4 + 3 * lam / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + lam)
            c_mu = min(1 - c_1, 2 * (mu_c - 2 + 1 / mu_c) / ((dim + 2) ** 2 + lam))
            # Weights for selection
            w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            w = w / np.sum(w)
            mu_eff = 1 / np.sum(w[:min(lam, int(lam / 2))] ** 2)

            # Local best for this restart
            local_best_x = x_mean.copy()
            local_best_f = func(local_best_x)

            calls_used = 1  # for the initial evaluation if restart > 0

            for gen in range(budget_per_restart - 1):
                if time.time() > deadline:
                    break
                if calls_used >= budget_per_restart:
                    break

                # Generate population
                xs = np.zeros((lam, dim))
                fs = np.zeros(lam)
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * np.linalg.cholesky(C) @ z
                    x = np.clip(x, -5, 5)
                    xs[i] = x
                    fx = func(x)
                    fs[i] = fx
                    calls_used += 1

                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()
                        if fx < global_best_f:
                            global_best_f = fx
                            global_best_x = x.copy()

                if calls_used >= budget_per_restart:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                xs_sorted = xs[idx]
                fs_sorted = fs[idx]

                # Update mean
                x_mean_new = np.sum(w[:, np.newaxis] * xs_sorted[:int(np.sum(w > 0))], axis=0)
                x_mean_new = np.clip(x_mean_new, -5, 5)

                # Update sigma (1/5 success rule approximation)
                # Use weighted step size
                step = np.linalg.norm(x_mean_new - x_mean)
                if step > 1e-10:
                    sigma = sigma * np.exp(c_sigma / mu_sigma * (step / sigma - 1))
                    sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance matrix
                y = (x_mean_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y

                C = (1 + c_1 * (mu_eff + 2) - c_1) * C
                C += c_1 * np.outer(y, y)
                C += c_mu * np.sum(w[:int(np.sum(w > 0))] * np.outer(xs_sorted[:int(np.sum(w > 0))] - x_mean, xs_sorted[:int(np.sum(w > 0))] - x_mean), axis=0) / sigma ** 2

                # Ensure positive semi-definite
                C = 0.5 * (C + C.T)
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(dim)

                x_mean = x_mean_new

            # After CMA-ES restart, if local best improved global, continue
            if local_best_f < global_best_f:
                global_best_f = local_best_f
                global_best_x = local_best_x.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
