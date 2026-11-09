"""CMA-ES for the BBOB-style Bent Cigar (f12) black-box task."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Full-matrix CMA-ES with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()
        DEADLINE = 300.0 - 20.0  # safety margin before the 330s kill

        rng = np.random.default_rng(12345)

        # --- CMA-ES constants ---
        mu = max(1, dim // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2.1) + c_sigma) / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1,
                   (2 * (mueff / (dim + 2) - 1) + c_sigma) /
                   ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # --- initialize ---
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        # precompute sqrt(C) = I
        sqrtC = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def make_population():
            z = rng.standard_normal((mu, dim))
            # y = sigma * (z @ sqrtC)  -> rows of (z @ sqrtC)
            y = (z @ sqrtC) * sigma
            return x_mean + y, z

        # initial population + evaluations
        pop, zpop = make_population()
        fvals = np.empty(mu)
        for i in range(mu):
            fvals[i] = func(np.clip(pop[i], -5.0, 5.0))

        best_x = np.clip(pop[np.argmin(fvals)], -5.0, 5.0).copy()
        best_f = float(fvals.min())
        used = mu
        gen = 0

        while used < budget:
            # ---- select & recombine ----
            order = np.argsort(fvals)
            y_sorted = np.array([pop[i] for i in order])
            z_sorted = np.array([zpop[i] for i in order])
            z_w = weights @ z_sorted
            y_w = weights @ y_sorted

            # ---- update evolution paths ----
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (sqrtC @ z_w)
            norm_ps = np.linalg.norm(ps) / chiN
            hsig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_w / sigma

            # ---- update covariance matrix (rank-mu + rank-1) ----
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C)
            C += c_mu * (weights @ np.einsum('ij,ik->jk', y_sorted, y_sorted) / sigma ** 2)

            # ---- update sigma (CSA) ----
            sigma *= np.exp(c_sigma / chiN * (norm_ps / chiN - 1))
            sigma = float(np.clip(sigma, 1e-15, 1e5))

            # ---- update mean ----
            x_mean = x_mean + y_w

            # ---- recompute sqrt(C) with stabilization ----
            try:
                C = 0.5 * (C + C.T)
                w, V = np.linalg.eigh(C)
                w = np.clip(w, 1e-20, None)
                sqrtC = V @ np.diag(np.sqrt(w))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                sqrtC = np.eye(dim)

            gen += 1
            used += mu

            # ---- next population ----
            if used >= budget:
                break
            if time.perf_counter() > DEADLINE:
                break
            pop, zpop = make_population()
            fvals = np.empty(mu)
            for i in range(mu):
                fvals[i] = func(np.clip(pop[i], -5.0, 5.0))
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = np.clip(pop[i], -5.0, 5.0).copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
