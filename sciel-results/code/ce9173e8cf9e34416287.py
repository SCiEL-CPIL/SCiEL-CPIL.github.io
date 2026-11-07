"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280

        rng = np.random.default_rng(42)

        # Phase 1: Find a good starting point
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1

        for _ in range(9):
            xi = rng.uniform(-5, 5, dim)
            fi = func(xi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
            calls += 1

        # Phase 2: CMA-ES
        mean = best_x.copy()
        sigma = 1.0
        lambda_pop = max(2 * dim + 2, 14)
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        gen = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Compute C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T

            # Sample
            z = rng.standard_normal((lambda_pop, dim))
            y = z @ C_inv_sqrt
            pop = mean[None, :] + sigma * y
            pop = np.clip(pop, -5, 5)

            # Evaluate
            pop_f = np.empty(lambda_pop)
            for i in range(lambda_pop):
                if calls >= budget or time.time() > deadline:
                    pop_f[i] = np.inf
                else:
                    pop_f[i] = func(pop[i])
                    calls += 1

            # Sort
            idx = np.argsort(pop_f)
            pop_s = pop[idx]
            pop_f_s = pop_f[idx]

            if pop_f_s[0] < best_f:
                best_f = pop_f_s[0]
                best_x = pop_s[0].copy()

            # Update mean
            new_mean = np.sum(weights[:, None] * pop_s[:mu], axis=0)

            # Path accumulators
            y_mean = C_inv_sqrt @ (new_mean - mean)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
            norm_ps = np.linalg.norm(ps)
            hsig = norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi
            hs = float(hsig)
            pc = (1 - c_c) * pc + hs * np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - mean) / sigma

            # Covariance update
            C = (1 + c_1 * (hs * (1 - c_c) - 1)) * C + c_1 * np.outer(pc, pc)
            # Symmetrize
            C = 0.5 * (C + C.T)

            # Sigma update
            sigma *= np.exp(c_sigma / chi * (norm_ps / chi - 1))
            sigma = np.clip(sigma, 1e-12, 20.0)

            mean = new_mean
            gen += 1

            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
