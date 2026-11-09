"""Rewritten optimizer for BBOB f1 (Sphere) with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f1 (Sphere) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        w = w / w.sum()
        mu_eff = 1.0 / (w[:mu] ** 2).sum()

        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 + 2 / (dim + 1) - (dim + 2) * c1) / ((dim + 2) ** 2 + 4))
        cs = (dim + 1.3) / (dim + 5 + 2)
        dsigma = (1 + 2 / max(1, dim - 1) + cmu / 2) * (1 + 2 / (dim + 1))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        mean = np.zeros(dim)
        sigma = 3.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)

        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1

        def time_exceeded():
            return time.time() > deadline

        while calls < budget:
            if time_exceeded():
                break

            # Generate and evaluate population
            pop = np.zeros((lam, dim))
            pop_f = np.zeros(lam)

            for i in range(lam):
                if time_exceeded() or calls >= budget:
                    break
                z = rng.standard_normal(dim)
                step = B @ (D * z)
                x_i = np.clip(mean + sigma * step, -5, 5)
                pop[i] = x_i
                pop_f[i] = func(x_i)
                calls += 1
                if pop_f[i] < best_f:
                    best_f = pop_f[i]
                    best_x = x_i.copy()

            if calls >= budget:
                break

            # Sort
            order = np.argsort(pop_f)
            pop_sorted = pop[order]

            # Update mean
            mean_new = np.zeros(dim)
            for i in range(mu):
                mean_new += w[i] * pop_sorted[i]

            # Update evolution paths
            delta_mean = (mean_new - mean) / sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * delta_mean
            ps = (1 - dsigma) * ps + np.sqrt(dsigma * (2 - dsigma) * mu_eff) * (C @ delta_mean)

            # Adapt sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            outer = np.zeros((dim, dim))
            for i in range(mu):
                outer += w[i] * np.outer(pop_sorted[i] - mean, pop_sorted[i] - mean)
            c_c = (1 + 2 * max(0, cs / c1 - 1) * (pc @ pc - 1)) * c1
            C = (1 - c1 - cmu + c_c) * C + c1 * np.outer(pc, pc) + (cmu / sigma ** 2) * outer

            # Eigendecomposition
            D_sq, B = np.linalg.eigh(C)
            D_sq = np.clip(D_sq, 1e-20, None)
            D = np.sqrt(D_sq)

            mean = mean_new

        return best_x, best_f
# EVOLVE-BLOCK-END
