"""Optimizer for BBOB-style f5 (Linear slope) on [-5,5]^5."""
import time
import numpy as np


class Optimizer:
    """CMA-ES with time guard for BBOB f5 minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        pop = max(4, int(round(4 + 3 * np.log(n))))
        w = np.log(pop / 2) - np.sort(np.log(np.arange(pop, 0, -1)))
        w /= w.sum()
        mu = int(w.nonzero()[0].size / 2)
        mueff = int(np.sum(w[:mu]) ** 2 / np.sum(w ** 2))
        c1 = (2 + 2 / (n + 1.3) ** 2) / ((n + 4) ** 1.7 + pop + 2)
        cmu = min(1 - c1, (2 * (mueff - 2 + 2 / mueff)) / ((n + 2) ** 1.6 + pop + 2))
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x = np.zeros(n)
        sigma = 2.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        D = np.ones(n)
        B = np.eye(n)

        def decomp(mat):
            eigvals, eigvecs = np.linalg.eigh(mat)
            eigvals = np.clip(eigvals, 1e-20, None)
            D_ = np.sqrt(eigvals)
            B_ = eigvecs @ np.diag(1.0 / D_)
            return D_, B_

        D, B = decomp(C)

        best_x = x.copy()
        best_f = np.inf
        calls = 0

        def eval_clip(v):
            nonlocal calls
            calls += 1
            return func(np.clip(v, -5, 5))

        # Initial population
        ys = []
        fvals = []
        for i in range(pop):
            y = x + sigma * B @ D * rng.standard_normal(n)
            ys.append(y)
            fvals.append(eval_clip(y))
        if min(fvals) < best_f:
            best_f = min(fvals)
            best_x = np.clip(ys[fvals.index(best_f)], -5, 5)

        gen = 0
        while calls < budget:
            if time.time() > deadline:
                break
            gen += 1
            order = np.argsort(fvals)
            sel = order[:mu]
            y_mean = np.mean([ys[i] for i in sel], axis=0)

            # Evolution paths
            diff = y_mean - x
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) / (D * (B.T @ diff))
            norm_pc = np.linalg.norm(pc)
            hsig = (norm_pc / np.sqrt(1 - (1 - cs) ** (gen + 1)) / chi < (1.4 + 2 / (n + 1)))
            ps = (1 - ds) * ps + (ds if hsig else 0) * np.sqrt(mueff) / D * (B.T @ diff)
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            sigma = max(sigma, 1e-12)

            # Covariance update
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cc * (1 - cc) if hsig else 0) * C)
            for i in range(mu):
                d = ys[sel[i]] - x
                C += cmu * w[i] * np.outer(d, d)
            C = 0.5 * (C + C.T)
            try:
                D, B = decomp(C)
            except Exception:
                C = np.eye(n)
                D, B = decomp(C)

            x = y_mean

            # New population
            ys = []
            fvals = []
            for i in range(pop):
                y = x + sigma * B @ D * rng.standard_normal(n)
                ys.append(y)
                fvals.append(eval_clip(y))
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = np.clip(y, -5, 5)
                if calls >= budget:
                    break

        return best_x, best_f
