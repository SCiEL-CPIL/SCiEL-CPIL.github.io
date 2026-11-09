"""CMA-ES optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with full covariance adaptation for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        lam = max(4, 4 + int(3 * np.log(n)))
        mu = max(1, lam // 2)

        weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        rng = np.random.default_rng(42)
        calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            fval = float(func(x_clipped))
            calls += 1
            if fval < best_f:
                best_f = fval
                best_x = x_clipped.copy()
            return fval

        # Initial population
        pop = []
        for i in range(lam):
            z = rng.standard_normal(n)
            x = m + sigma * z
            pop.append((evaluate(x), x))
        
        pop.sort(key=lambda t: t[0])

        while budget - calls >= lam:
            m_old = m.copy()
            m = m + sum(weights[i] * pop[i][1] for i in range(mu))

            delta_mean = (m - m_old) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * delta_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * delta_mean

            C = (1 + c_1 * (mueff - 1) - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc) \
                + c_mu * sum(weights[i] * np.outer(pop[i][1] - m_old, pop[i][1] - m_old) for i in range(mu))

            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)

            pop = []
            for i in range(lam):
                z = rng.standard_normal(n)
                x = m + sigma * (L @ z)
                pop.append((evaluate(x), x))

            pop.sort(key=lambda t: t[0])

        # Use remaining evaluations
        while calls < budget:
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
            z = rng.standard_normal(n)
            x = m + sigma * (L @ z)
            fval = evaluate(x)
            if fval < pop[0][0]:
                pop[0] = (fval, x)
                m = x.copy()
                sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
