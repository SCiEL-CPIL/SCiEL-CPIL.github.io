"""CMA-ES optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with covariance matrix adaptation for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of selected individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 1.3) ** 2 + mu_eff + 13)
        c_mu = min(1 - c_1,
                   (2 * (mu_eff / 2 - 1 + 2 / (dim + 1))) / ((dim + 2) ** 2 + mu_eff + 13))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def invsqrt(C):
            """Compute inverse square root of symmetric positive definite matrix."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Initial population
        pop = []
        for i in range(lambda_):
            z = rng.standard_normal(dim)
            x = np.clip(x_mean + sigma * z, -5, 5)
            fx = func(x)
            pop.append((fx, x, z))

        best_fx = min(p[0] for p in pop)
        calls_made = lambda_

        while calls_made < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Sort by fitness
            pop.sort(key=lambda p: p[0])

            # Update best
            if pop[0][0] < best_fx:
                best_fx = pop[0][0]

            # Selection: top mu individuals
            selected = pop[:mu]
            x_selected = np.array([p[1] for p in selected])
            z_selected = np.array([p[2] for p in selected])

            # Weighted mean
            x_new = np.sum(weights[:, None] * x_selected, axis=0)

            # Update evolution paths
            C_inv_sqrt = invsqrt(C)
            pc = (1 - c_c) * pc + c_c * C_inv_sqrt @ (x_new - x_mean) / sigma
            norm_pc = np.linalg.norm(pc)
            hsig = 1.0 if (norm_pc / np.sqrt(1 - (1 - c_c) ** (2 * (calls_made // lambda_ + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x_mean) / sigma

            # Covariance matrix update
            delta_hs = (1 - hsig) * c_sigma * (2 - c_sigma)
            rank_one = np.outer(ps, ps)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z_diff = C_inv_sqrt @ (x_selected[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(z_diff, z_diff)

            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_hs * C) + c_mu * rank_mu

            # Step size control
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update mean
            x_mean = x_new

            # Generate new population
            pop = []
            for i in range(lambda_):
                if time.time() > deadline - 5.0:
                    break
                z = rng.standard_normal(dim)
                x = np.clip(x_mean + sigma * C_inv_sqrt @ z, -5, 5)
                fx = func(x)
                calls_made += 1
                pop.append((fx, x, z))
                if fx < best_fx:
                    best_fx = fx

            if len(pop) < lambda_:
                break

        return x_mean, best_fx
# EVOLVE-BLOCK-END
