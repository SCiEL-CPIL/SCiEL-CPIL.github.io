# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Two-phase optimizer: coarse structured search + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(7)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_pt(x):
            nonlocal calls, best_f, best_x
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: structured exploration
        # Evaluate center
        eval_pt(np.zeros(dim))

        # Axis-aligned probes at multiple radii
        for i in range(dim):
            for r in (0.5, 1.0, 2.0, 3.0, 4.0, 4.5):
                for s in (1.0, -1.0):
                    pt = np.zeros(dim)
                    pt[i] = s * r
                    eval_pt(pt)

        # Face-center probes
        for i in range(dim):
            for s in (1.0, -1.0):
                pt = np.zeros(dim)
                pt[i] = s * 4.5
                eval_pt(pt)

        # Random points to fill exploration budget
        n_explore = min(budget // 4, 80)
        while calls < n_explore:
            pt = rng.uniform(-4.5, 4.5, dim)
            eval_pt(pt)

        # Phase 2: CMA-ES
        lam = 4 + int(np.floor(3 * np.log(dim)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        c1 = (2 + 2 / (dim + 1)) / ((dim + 1.3) ** 2 + 2)
        cmu = (2 + 2 / (dim + 1) - dim * c1) / ((dim + 1.3) ** 2 + 2)
        cs = (dim + 1.7) / (dim + 9.2)
        ds = 1 + 2 * max(0.0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        m = best_x.copy()
        C = np.eye(dim)
        sigma = 0.8
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        gen = 0

        while calls + lam <= budget:
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)

            off = np.empty((lam, dim))
            off_f = np.empty(lam)
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = m + sigma * (L @ z)
                off[i] = y
                off_f[i] = eval_pt(y)

            idx = np.argsort(off_f)
            off_s = off[idx]

            new_m = np.zeros(dim)
            for i in range(mu):
                new_m += w[i] * off_s[i]

            d_m = new_m - m
            try:
                d_scaled = np.linalg.solve(L.T, d_m / sigma)
            except np.linalg.LinAlgError:
                d_scaled = d_m / sigma

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * d_scaled

            denom = np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            if denom < 1e-12:
                denom = 1e-12
            hsig = (np.linalg.norm(ps) / denom) < (1.4 + 2 / (dim + 1)) * chi_n

            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * d_m / sigma

            C_new = (1 - c1 - cmu) * C
            C_new += c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (2 - cs) * np.outer(ps, ps))
            for i in range(mu):
                y_i = (off_s[i] - m) / sigma
                C_new += cmu * w[i] * np.outer(y_i, y_i)

            C = (C_new + C_new.T) / 2
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-9, 8.0)
            m = np.clip(new_m, -5.0, 5.0)
            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
