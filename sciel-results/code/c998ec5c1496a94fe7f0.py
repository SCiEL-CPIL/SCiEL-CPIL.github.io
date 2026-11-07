"""Optimized program for BBOB f6 (Attractive sector) with deadline guard."""
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 280s safety margin (limit is 300s)

        # Phase 1: Multi-start local search to find a good basin
        # Use several random starts with (1+1)-ES style local refinement
        best_x = None
        best_f = np.inf

        # Try a few diverse starting points
        rng = np.random.default_rng(42)
        starts = []
        # Center and a few random points
        starts.append(np.zeros(dim))
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))

        calls_used = 0
        max_calls_phase1 = min(budget // 3, 200)

        for start in starts:
            if calls_used >= max_calls_phase1:
                break
            if time.time() > deadline:
                break

            x = start.copy()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Local (1+1)-ES from this start
            sigma = 1.0
            local_budget = max_calls_phase1 - calls_used
            for _ in range(local_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                if sigma < 1e-6:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: CMA-ES from the best point found
        remaining = budget - calls_used
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~12 for dim=5
        pop_size = min(pop_size, max(4, remaining // 5))
        if pop_size < 2:
            pop_size = 2

        x_best = best_x.copy()
        f_best = best_f
        sigma = 0.5
        mean = x_best.copy()

        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Adaptation parameters
        c_sigma = (pop_size + 2) / (dim + 4)
        c_c = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size // 2 + 1) - dim / 2) / ((dim + 2) ** 2 + pop_size))
        mu_lambda = pop_size // 2
        weights = np.log(mu_lambda + 1) - np.log(np.arange(1, mu_lambda + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        c_cov = (2 + 2 * mu_eff / (dim + 1)) / ((dim + 2) ** 2 + mu_eff)
        d_cov = 1 + c_cov * (c_1 - c_cov) + np.sqrt(
            (mu_eff / (dim + 1) - 1) * (c_1 - c_cov) * (2 - c_cov / c_1)
        )

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Track best
        archive_best_x = best_x.copy()
        archive_best_f = best_f

        cma_calls = 0
        max_cma_calls = remaining

        while cma_calls < max_cma_calls:
            if time.time() > deadline:
                break

            # Generate population
            samples = []
            for i in range(pop_size):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                samples.append(y)

            # Evaluate
            fvals = np.array([func(s) for s in samples])
            cma_calls += pop_size

            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < archive_best_f:
                archive_best_f = fvals[idx_min]
                archive_best_x = samples[idx_min].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = [samples[i] for i in order]
            f_sorted = fvals[order]

            # Weighted mean
            y_mean = np.zeros(dim)
            for i in range(mu_lambda):
                y_mean += weights[i] * y_sorted[i]

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * (C_sqrt_inv @ ((sigma * (y_mean - mean)) / sigma))

            # Hessian normalization
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = int(norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (cma_calls + 1))) / chi_n < 1.4 + 2 / (dim + 1))

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(
                c_c * (2 - c_c) * mu_eff
            ) * (C_sqrt_inv @ ((y_mean - mean) / sigma))

            # Update mean
            mean = y_mean

            # Update covariance
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu_lambda):
                diff = (y_sorted[i] - y_mean) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)

            C = (1 + c_1 * (h_sigma - 1) + c_cov) * C - c_1 * rank_1 - c_cov * rank_mu

            # Update sigma
            sigma *= np.exp(
                c_sigma / chi_n * (norm_p_sigma / chi_n - 1)
            )
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update C_sqrt (approximate via eigendecomposition periodically)
            # Do it every 10 iterations for efficiency
            if cma_calls % (10 * pop_size) < pop_size:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2

            # If sigma is very small, we've converged
            if sigma < 1e-8:
                break

        # Phase 3: Final polish with very fine local search
        polish_calls = min(20, budget - calls_used - cma_calls)
        if polish_calls > 0 and time.time() < deadline:
            x = archive_best_x.copy()
            fx = archive_best_f
            sigma = 0.01
            for _ in range(polish_calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7
                if sigma < 1e-9:
                    break
            if fx < archive_best_f:
                archive_best_f = fx
                archive_best_x = x.copy()

        return archive_best_x, archive_best_f
