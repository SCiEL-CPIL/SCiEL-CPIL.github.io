"""Optimizer for BBOB-style f15 (Rastrigin rotated) using CMA-ES with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    with multi-start strategy for robustness against multimodality.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # CMA-ES implementation
        def cma_es(start_x, max_iter, sigma0):
            nonlocal calls_made, best_x, best_f, deadline

            if calls_made >= budget or time.time() > deadline:
                return

            # Parameters
            pop_size = 4 + int(3 * np.log(dim))  # 4 + 3*ln(5) ≈ 8
            pop_size = max(4, min(pop_size, 20))

            # CMA-ES state
            m = start_x.copy()
            sigma = sigma0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            sqrtC = np.eye(dim)
            sqrtC_inv = np.eye(dim)

            # Update rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[1] - 1) / ((dim + 2) ** 2 + mu_eff))

            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            # Initial evaluation
            f_m = evaluate(m)
            f_best_local = f_m
            x_best_local = m.copy()

            # Precompute
            for it in range(max_iter):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Generate population
                ys = []
                xs = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    y = z
                    x = m + sigma * (sqrtC @ y)
                    x = np.clip(x, -5, 5)
                    ys.append(y)
                    xs.append(x)

                # Evaluate population
                fs = np.array([evaluate(x) for x in xs])

                # Sort
                sorted_idx = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in sorted_idx])
                xs_sorted = np.array([xs[i] for i in sorted_idx])
                fs_sorted = fs[sorted_idx]

                # Update best
                if fs_sorted[0] < f_best_local:
                    f_best_local = fs_sorted[0]
                    x_best_local = xs_sorted[0].copy()

                # Update mean
                m_new = m + sigma * (sqrtC @ (np.sum(weights[:, None] * ys_sorted[:mu], axis=0)))
                m = m_new

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (sqrtC_inv @ (np.sum(weights[:, None] * ys_sorted[:mu], axis=0)))
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)

                # Update C
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (sqrtC_inv @ (np.sum(weights[:, None] * ys_sorted[:mu], axis=0)))

                delta_heaviside = 1.0 if (np.linalg.norm(p_c) ** 2) / ((1 - (1 - c_c) ** (2 * (it + 1))) * dim) < (2 + 2.0 / (dim + 1)) * chi_n ** 2 else 0.0

                C = (1 + c_1 * delta_heaviside - c_1 - c_mu) * C + \
                    c_1 * (np.outer(p_c, p_c) + (delta_heaviside - 1) * C) + \
                    c_mu * np.sum(weights[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu))

                # Ensure C is symmetric
                C = (C + C.T) / 2

                # Update sqrtC via SVD (approximate)
                # For simplicity, use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                sqrtC_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Track global best
                if f_best_local < best_f:
                    best_f = f_best_local
                    best_x = x_best_local.copy()

            return x_best_local, f_best_local

        # Multi-start strategy
        # First, do some random sampling to find good starting points
        n_initial = min(20, budget // 10)
        for i in range(n_initial):
            if calls_made >= budget or time.time() > deadline:
                break
            if i == 0:
                x_try = np.zeros(dim)
            else:
                x_try = rng.uniform(-4, 4, dim)
            evaluate(x_try)

        # Now run CMA-ES from the best point found
        if best_x is not None and best_f < float('inf'):
            remaining_budget = budget - calls_made
            if remaining_budget > 50:
                max_iter = min(remaining_budget // 8, 200)  # Approximate iterations
                cma_es(best_x.copy(), max_iter, 1.0)

        # If we still have budget, try another CMA-ES from a different point
        if calls_made < budget - 50 and time.time() < deadline:
            # Try a perturbation of the best point
            x_start2 = best_x + 0.5 * rng.standard_normal(dim)
            x_start2 = np.clip(x_start2, -5, 5)
            remaining_budget = budget - calls_made
            if remaining_budget > 50:
                max_iter = min(remaining_budget // 8, 150)
                cma_es(x_start2, max_iter, 0.5)

        # Final polish with (1+1)-ES
        if calls_made < budget and time.time() < deadline:
            current_x = best_x.copy()
            current_f = best_f
            sigma = 0.1
            window = 10
            successes = []

            while calls_made < budget and time.time() < deadline:
                y = current_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)

                if fy < current_f:
                    current_x = y
                    current_f = fy
                    successes.append(1)
                else:
                    successes.append(0)

                if len(successes) >= window:
                    success_rate = np.mean(successes)
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    sigma = max(sigma, 1e-4)
                    successes = []

        return best_x, best_f
# EVOLVE-BLOCK-END
