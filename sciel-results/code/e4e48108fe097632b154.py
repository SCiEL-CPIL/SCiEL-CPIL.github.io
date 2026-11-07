"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard and restarts for rotated 5D landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_ = max(4 * n, 10)  # population size
        mu = lambda_ // 2  # number of selected solutions
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (2 - cc)
        cmu = min(1 - c1, (2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)) * (2 - cc))

        def cma_es(x0, sigma0, remaining_budget, rng_local):
            x = x0.copy()
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            C_sqrt = np.eye(n)  # We'll use Cholesky or eigendecomp each step for simplicity
            # Actually, for 5D, we can just use the matrix directly
            # Generate initial population
            best_x = x0.copy()
            best_f = func(x0)
            calls = 1
            gen = 0
            while calls < remaining_budget:
                if time.time() > deadline - 2:
                    break
                gen += 1
                # Generate population
                ys = []
                fs = []
                for i in range(lambda_):
                    if calls >= remaining_budget:
                        break
                    if time.time() > deadline - 2:
                        break
                    # Generate step
                    z = rng_local.standard_normal(n)
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fs.append(fy)
                    calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(ys) < mu:
                    break

                # Sort by fitness
                sorted_idx = np.argsort(fs)
                ys = [ys[i] for i in sorted_idx[:mu]]
                fs = [fs[i] for i in sorted_idx[:mu]]

                # Update mean
                x_old = x.copy()
                x = sum(w * y for w, y in zip(weights, ys))
                x = np.clip(x, -5, 5)

                # Update sigma
                # Compute C_sqrt_inv * (x - x_old)
                # For simplicity in 5D, use eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                d = x - x_old
                C_sqrt_inv_d = C_sqrt_inv @ d
                norm_d = np.linalg.norm(C_sqrt_inv_d)
                if norm_d > 1e-12:
                    hsig = (1 - np.prod(1 - cc) ** (1 / cs)) / np.sqrt(1 - np.prod(1 - cc) ** (2 / cs))
                    p_sigma = (1 - cs) * p_sigma + cs * (norm_d / np.sqrt(2) - 1) * C_sqrt_inv_d / norm_d
                    sigma *= np.exp(cs / (n + 2) * (norm_d / np.sqrt(2) - 1))
                    sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C
                # Compute rank-1 update
                diff = x - x_old
                C = (1 + c1 * cc) * C + c1 * np.outer(diff, diff)

                # Compute rank-mu update
                # Use cumulative step vectors
                y_diffs = [y - x_old for y in ys]
                # Transform by C_sqrt_inv
                y_diffs_transformed = [C_sqrt_inv @ yd for yd in y_diffs]
                # Weighted sum
                h = np.zeros(n)
                for w, ydt in zip(weights, y_diffs_transformed):
                    h += w * ydt
                # Rank-mu update
                C += cmu * np.outer(h, h)

                # Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T

                # Update p_c
                hsig = (1 - np.prod(1 - cc) ** (1 / cs)) / np.sqrt(1 - np.prod(1 - cc) ** (2 / cs))
                p_c = (1 - cc) * p_c + cc * hsig * (norm_d / np.sqrt(2)) * C_sqrt_inv_d / (norm_d + 1e-12) * (norm_d / np.sqrt(2))
                # Actually, let's simplify p_c update
                p_c = (1 - cc) * p_c + cc * C_sqrt_inv_d / (norm_d + 1e-12) * (norm_d / np.sqrt(2))

                # Update C with p_c
                C = (1 - c1 * cc * (2 - cc)) * C + c1 * (1 + cc * (2 - cc)) * np.outer(p_c, p_c)

                # Ensure C is positive definite again
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T

            return best_x, best_f, calls

        # Multi-start CMA-ES
        best_x = None
        best_f = np.inf
        total_calls = 0

        # First, do a quick random search to find a good starting point
        n_random = min(20, budget // 5)
        for i in range(n_random):
            if total_calls >= budget:
                break
            if time.time() > deadline - 5:
                break
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            total_calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        # Now run CMA-ES from the best starting point
        remaining = budget - total_calls
        if remaining > 10:
            x_opt, f_opt, calls_used = cma_es(best_x, 2.0, remaining, rng)
            total_calls += calls_used
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # If we have budget left and time, do another CMA-ES restart from a different point
        if total_calls < budget and time.time() < deadline - 30:
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            total_calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

            remaining = budget - total_calls
            if remaining > 10:
                x_opt, f_opt, calls_used = cma_es(best_x, 1.5, remaining, rng)
                total_calls += calls_used
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
