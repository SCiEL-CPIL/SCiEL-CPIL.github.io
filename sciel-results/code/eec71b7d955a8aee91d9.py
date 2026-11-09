"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES continuous optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point at origin
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        sigma = 2.0
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + 2)
        cmu = min(1 - c1, (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + 2))
        cs = (dim + 1.7) / (dim + 9)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # State
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        C = np.eye(dim)
        mueff = None

        # Initial population
        pop = np.array([np.clip(x + sigma * rng.standard_normal(dim), -5, 5) for _ in range(lam)])
        fvals = np.array([func(p) for p in pop])
        calls += lam
        order = np.argsort(fvals)
        pop = pop[order]
        fvals = fvals[order]
        x = pop[0].copy()
        fx = fvals[0]

        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        while calls < budget:
            # Compute new mean
            m = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update evolution paths
            C_inv = np.linalg.inv(C + 1e-10 * np.eye(dim))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv @ (m - x)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * max(1, budget - calls)))) < (1.4 + 2 / (dim + 1)) * chi
            hs = 1.0 if hsig else 0.0
            pc = (1 - c1) * pc + hs * np.sqrt(c1 * (2 - c1) * mueff) * np.outer(m - x, m - x) / (sigma * sigma)

            # Update covariance
            delta_h = (1 - hs) * c1 * (2 - c1)
            C = (1 - c1 - cmu + delta_h) * C
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(pop[i] - x, pop[i] - x) / (sigma * sigma)
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Update mean
            x = m

            # Generate new population
            new_pop = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                C_inv_new = np.linalg.inv(C + 1e-10 * np.eye(dim))
                y = x + sigma * (C_inv_new @ z)
                y = np.clip(y, -5, 5)
                new_pop.append(y)
            new_pop = np.array(new_pop)

            # Evaluate
            new_fvals = np.array([func(p) for p in new_pop])
            calls += lam

            # Sort
            order = np.argsort(new_fvals)
            new_pop = new_pop[order]
            new_fvals = new_fvals[order]

            if new_fvals[0] < fx:
                fx = new_fvals[0]
                x = new_pop[0].copy()

            pop = new_pop
            fvals = new_fvals

        return x, fx
# EVOLVE-BLOCK-END
