"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        calls_used = 0

        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def check_time():
            return time.time() > deadline or calls_used >= total_budget

        def cma_es(x0, sigma0, budget_left, rng):
            nonlocal calls_used, best_x, best_f, deadline
            n = dim
            x = x0.copy()
            fx = func(x)
            track_call(x, fx)
            sigma = sigma0

            # CMA-ES parameters
            lambda_ = 4 * n  # population size
            mu = lambda_ // 2  # number of recombination weights
            # Weights
            weights = np.log(np.arange(mu, 0, -1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            # Step size control parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            # Covariance matrix adaptation
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            # Evolution path parameters
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # Initialize
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            inv_sqrt_C = np.eye(n)
            dcor = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma

            for _ in range(budget_left):
                if check_time():
                    return

                # Generate population
                Y = []
                F = []
                for i in range(lambda_):
                    if check_time():
                        return
                    z = rng.standard_normal(n)
                    y = np.clip(x + sigma * (inv_sqrt_C @ z), -5, 5)
                    fy = func(y)
                    track_call(y, fy)
                    Y.append(y)
                    F.append(fy)

                # Sort by fitness
                idx = np.argsort(F)
                Y = [Y[i] for i in idx]
                F = [F[i] for i in idx]

                # Update best
                if F[0] < fx:
                    fx = F[0]
                    x = Y[0].copy()

                # Mean recombination
                x_new = sum(weights[i] * Y[i] for i in range(mu))

                # Update evolution paths
                diff = (x_new - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * diff
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, calls_used % 100)))) < (1.4 + 2 / (n + 1)) * chi_n

                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * diff

                # Covariance matrix update
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_c * (c_1 + c_mu - 1) - c_c) * C
                C += c_1 * (np.outer(p_c, p_c) + delta_hsig * C)
                C += c_mu * sum(weights[i] * np.outer(Y[i] - x, Y[i] - x) for i in range(mu))

                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = (eigvecs * eigvals) @ eigvecs.T

                # Update inverse square root
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.sqrt(np.maximum(eigvals, 1e-20))
                inv_sqrt_C = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x = x_new

            return x, fx

        # Strategy:
        # 1. Brief random exploration to find a good starting region
        # 2. CMA-ES from the best point
        # 3. If budget remains, restart CMA-ES from a perturbed best point

        # Phase 1: Exploration (5% of budget)
        explore_budget = max(5, total_budget // 20)
        rng = np.random.default_rng(42)

        for _ in range(explore_budget):
            if check_time():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        # Phase 2: CMA-ES from best point (70% of remaining)
        remaining = total_budget - calls_used
        if remaining > 20:
            cma_budget = int(remaining * 0.7)
            cma_es(best_x, 1.0, cma_budget, rng)

        # Phase 3: If budget remains, do a short CMA-ES restart from perturbed best
        remaining = total_budget - calls_used
        if remaining > 10 and best_x is not None:
            x2 = best_x + rng.uniform(-0.5, 0.5, dim)
            x2 = np.clip(x2, -5, 5)
            cma_es(x2, 0.5, remaining, rng)

        # Phase 4: Final polish with very small steps
        remaining = total_budget - calls_used
        if remaining > 5 and best_x is not None:
            for _ in range(remaining):
                if check_time():
                    break
                step = 0.01 * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5, 5)
                fy = func(y)
                track_call(y, fy)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
