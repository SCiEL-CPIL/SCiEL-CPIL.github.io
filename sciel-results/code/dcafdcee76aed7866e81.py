"""Optimizer for BBOB f21 (Gallagher 101 peaks) using multi-start CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with restarts for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Multi-start strategy: use budget across multiple CMA-ES runs
        # Each run gets a portion of the budget
        n_starts = 8
        calls_per_start = budget // n_starts
        calls_per_start = max(calls_per_start, 50)

        def cma_es(x0, sigma0, max_calls):
            """Run CMA-ES from x0 with initial step size sigma0."""
            nonlocal best_x, best_f

            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))

            # CMA-ES parameters
            c_sigma = (n + 2) / (4 * n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + lam / n) / (4 + n + 3)
            c_1 = 2 / ((n + 1.3) ** 2 + lam)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((n + 2) * (lam - mu_eff + 0.5)) - 1))

            # Weights for recombination
            weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            weights /= weights.sum()
            mu = int(np.sum(weights > 0))
            mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

            # State
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            calls = 0

            for gen in range(max_calls // lam + 1):
                if time.time() > deadline:
                    break
                if calls >= max_calls:
                    break

                # Generate population
                for i in range(lam):
                    if calls >= max_calls or time.time() > deadline:
                        break
                    wi = weights[i]
                    xi = x_mean + sigma * (B @ (D * (wi * rng.standard_normal(n))))
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls += 1

                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                if calls >= max_calls:
                    break

                # Sort by fitness
                # Re-evaluate or keep track - we need to store
                # Let's restructure: store all evaluations

                break  # This approach needs restructuring

            return best_x, best_f

        def cma_es_proper(x0, sigma0, max_calls):
            """Proper CMA-ES implementation."""
            nonlocal best_x, best_f

            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))

            # Parameters
            c_sigma = (n + 2) / (4 * n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + lam / n) / (4 + n + 3)
            c_1 = 2 / ((n + 1.3) ** 2 + lam)

            # Weights
            weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            weights = np.maximum(weights, 0)
            if weights.sum() == 0:
                weights = np.ones(lam) / lam
            else:
                weights /= weights.sum()
            mu = int(np.sum(weights > 0))
            if mu < 1:
                mu = 1
            weights = weights[:mu]
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((n + 2) * (lam - mu_eff + 0.5)) - 1))

            # State
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            calls = 0

            while calls < max_calls:
                if time.time() > deadline:
                    break

                # Generate and evaluate population
                X = np.zeros((lam, n))
                F = np.zeros(lam)

                for i in range(lam):
                    if calls >= max_calls or time.time() > deadline:
                        break
                    xi = x_mean + sigma * (B @ (D * (weights[i] * rng.standard_normal(n))))
                    xi = np.clip(xi, -5, 5)
                    X[i] = xi
                    F[i] = func(xi)
                    calls += 1

                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = xi.copy()

                if calls >= max_calls:
                    break

                # Sort indices by fitness
                order = np.argsort(F)
                X = X[order]
                F = F[order]

                # Update mean
                x_new = np.sum(weights[:, None] * X[:mu], axis=0)

                # Update covariance matrix
                diff = (X[:mu] - x_mean) / sigma
                C_update = np.sum(weights[:, None, None] * np.outer(diff, diff), axis=0)
                C = B @ (D ** 2) @ B.T
                C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (np.dot(pc, pc) > c_c * (4 + n / n) - 1) * c_c * (2 - c_c) * C) + c_mu * C_update

                # Eigen decomposition
                D_sq, B = np.linalg.eigh(C)
                D_sq = np.maximum(D_sq, 1e-20)
                D = np.sqrt(D_sq)

                # Update step size
                ps = (1 - c_sigma) * ps + c_sigma / np.sqrt(c_sigma * (2 - c_sigma)) * (B @ (D * (x_new - x_mean) / sigma))
                chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50)

                # Update path
                pc = (1 - c_c) * pc + np.sqrt(c_c * (4 - c_c) * mu_eff) * (x_new - x_mean) / sigma

                x_mean = x_new

            return best_x, best_f

        # Phase 1: Broad exploration with multiple starts
        for s in range(n_starts):
            if time.time() > deadline:
                break
            if budget - (calls_per_start * s) <= 0:
                break

            # Random starting point
            x0 = rng.uniform(-5, 5, dim)
            sigma0 = 2.0 if s == 0 else 1.5

            remaining = budget - calls_per_start * s
            alloc = min(calls_per_start, remaining)

            if alloc < 10:
                break

            cma_es_proper(x0, sigma0, alloc)

        # Phase 2: Refinement around best found point
        if best_x is not None and time.time() < deadline:
            used_calls = budget - (n_starts * calls_per_start)
            if used_calls < 0:
                used_calls = 0
            remaining = budget - (budget - used_calls)  # This is just used_calls
            # Actually, let's track calls properly
            # For now, do a final refinement with remaining budget
            refinement_calls = min(100, max(0, budget - n_starts * calls_per_start))
            if refinement_calls > 0:
                cma_es_proper(best_x, 0.5, refinement_calls)

        return best_x, best_f
# EVOLVE-BLOCK-END
