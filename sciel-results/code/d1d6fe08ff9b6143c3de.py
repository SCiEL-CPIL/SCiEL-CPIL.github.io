"""Rewritten optimizer for BBOB f14 (Different Powers)."""
# EVOLVE-BLOCK-START
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
        deadline = time.time() + 280  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        c_sigma = (mu_eff + 1) / (dim + 2) / 2
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (sqrt(mu_eff) - 1) / ((dim + 2) ** 2 + mu_eff))

        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        def cma_es(x0, fx0, max_evals, start_sigma=1.0):
            """Run CMA-ES from x0 with max_evals evaluations."""
            x = x0.copy()
            fx = fx0
            sigma = start_sigma
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            c_sigma = (mu_eff + 1) / (dim + 2) / 2
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (np.sqrt(mu_eff) - 1) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            evals_used = 0
            best_x = x.copy()
            best_fx = fx

            while evals_used < max_evals:
                if time.time() > deadline:
                    break

                # Generate lambda offspring
                offspring = []
                for i in range(lambda_):
                    z = np.random.standard_normal(dim)
                    y = x + sigma * z @ np.linalg.cholesky(C + 1e-12 * np.eye(dim)).T
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals_used += 1
                    offspring.append((fy, y))
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()

                # Sort offspring
                offspring.sort(key=lambda t: t[0])

                # Update mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * offspring[i][1]

                # Update step size
                pc = (1 - c_c) * pc + c_c * (x_new - x) / sigma
                hsig = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_new - x) / sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update covariance
                delta = (x_new - x) / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (2 * np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * np.eye(dim)) \
                    + c_mu * np.sum(weights[:mu, None] * np.outer(delta, delta), axis=0)

                x = x_new

            return best_x, best_fx, evals_used

        # Multi-start strategy
        # Phase 1: Explore with several random starts and small CMA-ES runs
        n_starts = 5
        evals_per_start = max(20, (budget - n_starts) // n_starts)
        remaining = budget

        best_x = None
        best_fx = np.inf
        total_evals = 0

        rng = np.random.default_rng(42)

        for start_i in range(n_starts):
            if total_evals >= budget or time.time() > deadline:
                break

            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            total_evals += 1

            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()

            max_evals_this = min(evals_per_start, budget - total_evals)
            if max_evals_this < 10:
                break

            bx, bfx, used = cma_es(x0, fx0, max_evals_this, start_sigma=2.0)
            total_evals += used

            if bfx < best_fx:
                best_fx = bfx
                best_x = bx.copy()

        # Phase 2: Refine around the best point found
        if best_x is not None and total_evals < budget:
            remaining_evals = budget - total_evals
            if remaining_evals > 10:
                bx, bfx, used = cma_es(best_x, best_fx, remaining_evals, start_sigma=0.5)
                total_evals += used
                if bfx < best_fx:
                    best_fx = bfx
                    best_x = bx.copy()

        # Phase 3: If still have budget, do a final local refinement
        if best_x is not None and total_evals < budget:
            remaining_evals = budget - total_evals
            if remaining_evals > 5:
                bx, bfx, used = cma_es(best_x, best_fx, remaining_evals, start_sigma=0.1)
                total_evals += used
                if bfx < best_fx:
                    best_fx = bfx
                    best_x = bx.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
