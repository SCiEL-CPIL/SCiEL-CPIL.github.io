"""CMA-ES optimizer for BBOB f12 (bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time/budget guard for box-constrained continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0          # safety margin under 300s wall-clock
        budget = self.budget
        dim = self.dim

        # --- initial population (lambda evaluations) ---
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        cmu = w.sum() ** 2 / (w ** 2).sum()
        c_sigma = (mu + 0.3) / (dim + mu + 3)
        c_c    = (4 + cmu / dim) / (dim + 4 + 2 * cmu / dim)
        sigma0 = 2.0

        x = np.zeros(dim)
        fx = func(x)
        best_x, best_f = x.copy(), fx
        calls = 1

        if calls >= budget:
            return best_x, best_f

        # initial population around origin
        P = np.eye(dim)
        sigma = sigma0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # generate initial lambda samples (except the center which is x)
        Z = np.random.default_rng(0).standard_normal((lam, dim))
        pop = x + sigma * (Z @ P)
        pop = np.clip(pop, -5, 5)
        fvals = np.empty(lam)
        fvals[0] = fx
        for i in range(1, lam):
            if time.time() > deadline or calls >= budget:
                break
            fvals[i] = func(pop[i])
            calls += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = pop[i].copy()
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        # if we broke early, fill remaining fvals with inf
        fvals[np.isnan(fvals)] = np.inf
        if np.isnan(fvals).any():
            fvals[np.isnan(fvals)] = 1e300

        while calls < budget:
            if time.time() > deadline:
                break

            # sort by fitness
            order = np.argsort(fvals)
            y_sorted = pop[order]
            f_sorted = fvals[order]

            # update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # weighted mean of mu best
            y_w = (w[:, None] * y_sorted[:mu]).sum(axis=0)

            # sigma path update
            pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * cmu) / (sigma) * (P @ y_w)
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            cs = c_sigma * (1.0 if hsig else 0.0)
            sigma = sigma * np.exp(cs * (np.linalg.norm(pc) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # covariance path update
            pc = (1 - c_c) * pc + c_c * (y_w)
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                yi = y_sorted[i]
                rankmu += w[i] * np.outer(yi, yi)
            C = (1 + mu * c_c * c_c - c_c) * C + c_c * (rank1 - C) + (1 - c_c) * cmu * rankmu
            # re-symmetrize and ensure PSD via eigen clip
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = (eigvecs * eigvals) @ eigvecs.T
            C = (C + C.T) / 2
            P = np.linalg.cholesky(C + 1e-12 * np.eye(dim))

            # generate next population
            Z = np.random.default_rng().standard_normal((lam, dim))
            pop = x + sigma * (Z @ P)
            pop = np.clip(pop, -5, 5)
            fvals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline or calls >= budget:
                    fvals[i] = np.inf
                    continue
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
