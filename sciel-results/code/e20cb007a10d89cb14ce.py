"""Optimizer for BBOB-style f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES variant with restarts and adaptive population for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf

        # We'll do multiple short CMA-ES runs with different starting points
        # to handle the multimodal nature of Büche-Rastrigin

        def cma_es(x0, f0, n_calls_remaining, seed):
            nonlocal best_x, best_f, deadline
            rng = np.random.default_rng(seed)

            # Adaptive population size
            n = max(4, int(4 + 3 * np.log(dim)))
            # Initial step size
            sigma = 1.0
            x = x0.copy().astype(float)
            fx = f0

            # CMA-ES parameters
            cm = np.eye(dim)
            # Covariance matrix update parameters
            mu = n // 2  # number of weighted parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

            # State vectors
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            calls_used = 0

            while calls_used < n_calls_remaining:
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(n):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    # Evolve covariance (simplified: use Cholesky of cm)
                    try:
                        L = np.linalg.cholesky(cm)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                    y = sigma * L @ z
                    xi = x + y
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls_used += 1
                    offspring.append((fi, xi, y))

                if not offspring:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < fx:
                    fx = offspring[0][0]
                    x = offspring[0][1].copy()
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # Weighted mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * offspring[i][1]

                # Update evolution path for sigma
                y_diff = (x_new - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_diff

                # Update evolution path for covariance
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * cm @ y_diff

                # Update covariance matrix
                delta_sigma = (np.linalg.norm(ps) / chi_n - 1) * c_sigma * d_sigma
                cm = (1 + c_1 * delta_sigma) * cm
                cm -= c_1 * np.outer(pc, pc)
                for i in range(mu):
                    cm += c_mu * weights[i] * np.outer(offspring[i][2], offspring[i][2])

                # Ensure positive semi-definite
                try:
                    eigvals, eigvecs = np.linalg.eigh(cm)
                    eigvals = np.maximum(eigvals, 1e-12)
                    cm = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    cm = np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update mean
                x = x_new

                # If sigma is too small, break
                if sigma < 1e-12:
                    break

            return calls_used

        # Phase 1: Explore with multiple restarts
        n_starts = 8
        calls_per_start = max(50, budget // (n_starts + 1))
        total_calls = 1

        # First, try a few random points to get a sense of the landscape
        rng_init = np.random.default_rng(42)
        for i in range(min(5, budget - 1)):
            if time.time() > deadline:
                break
            x_try = rng_init.uniform(-5, 5, dim)
            f_try = func(x_try)
            total_calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        # Phase 2: CMA-ES from best found and other promising starts
        start_points = [best_x.copy()] if best_x is not None else [np.zeros(dim)]

        # Add some diverse starting points
        rng_starts = np.random.default_rng(123)
        for i in range(n_starts - 1):
            if time.time() > deadline:
                break
            # Start near the best point but with some perturbation
            if best_x is not None:
                sp = best_x + rng_starts.standard_normal(dim) * 1.5
            else:
                sp = rng_starts.uniform(-5, 5, dim)
            sp = np.clip(sp, -5, 5)
            start_points.append(sp)

        for i, sp in enumerate(start_points):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break

            # Evaluate starting point if not already done
            if i == 0 and best_x is not None:
                f_start = best_f
            else:
                f_start = func(sp)
                total_calls += 1
                if f_start < best_f:
                    best_f = f_start
                    best_x = sp.copy()

            remaining = budget - total_calls
            if remaining < 10:
                break

            # Use decreasing budget for later restarts
            alloc = min(remaining, calls_per_start)
            cma_es(sp, f_start, alloc, seed=1000 + i)
            total_calls += alloc

            if total_calls >= budget:
                break

        # Phase 3: Fine-tuning with small steps from best
        if best_x is not None and total_calls < budget:
            rng_fine = np.random.default_rng(999)
            sigma_fine = 0.5
            for _ in range(budget - total_calls):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma_fine * rng_fine.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma_fine *= 1.3
                else:
                    sigma_fine *= 0.85
                    if sigma_fine < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
