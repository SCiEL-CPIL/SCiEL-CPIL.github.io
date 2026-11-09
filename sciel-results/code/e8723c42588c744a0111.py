"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        rng = np.random.default_rng(42)

        # State
        x_mean = rng.uniform(-3, 3, n)
        sigma = 2.0
        C = np.eye(n)
        evals, evecs = np.linalg.eigh(C)
        C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        ps = np.zeros(n)
        pc = np.zeros(n)

        calls_made = 0

        def do_call(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx

        # Initial population
        z = rng.standard_normal((lambda_, n))
        pop = x_mean + sigma * (z @ C_inv_sqrt.T)
        pop = np.clip(pop, -5, 5)
        fvals = np.array([do_call(x) for x in pop])

        def cma_step(pop, fvals):
            nonlocal x_mean, sigma, C, C_inv_sqrt, ps, pc
            order = np.argsort(fvals)
            fvals_sorted = fvals[order]
            pop_sorted = pop[order]

            y_w = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            x_new = x_mean + sigma * y_w
            ps = np.sqrt(1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma)) / sigma * y_w
            ps_norm = np.linalg.norm(ps)

            if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 / budget)) < (1.4 + 1 / n) * chi_n:
                pc = (1 - c_c) * pc + c_c * (x_new - x_mean) / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * np.sum(
                    weights[:, None, None] * np.outer(pop_sorted[:mu] - x_mean, pop_sorted[:mu] - x_mean) / sigma ** 2,
                    axis=0
                )
                C = (C + C.T) / 2
                sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))

            x_mean = x_new

            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.maximum(evals, 1e-12)
                C = evecs @ np.diag(evals) @ evecs.T
                C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_inv_sqrt = np.eye(n)

        cma_step(pop, fvals)

        # Continue until budget exhausted
        while calls_made < budget:
            z = rng.standard_normal((lambda_, n))
            pop = x_mean + sigma * (z @ C_inv_sqrt.T)
            pop = np.clip(pop, -5, 5)
            fvals = np.array([do_call(x) for x in pop])
            cma_step(pop, fvals)

        return x_mean, fvals.min()

# EVOLVE-BLOCK-END
