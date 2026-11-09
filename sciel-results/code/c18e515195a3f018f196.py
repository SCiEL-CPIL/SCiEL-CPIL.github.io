"""Optimized continuous optimizer for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES for minimizing f on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ------------------------------------------------------------------ #
    # CMA-ES core
    # ------------------------------------------------------------------ #
    def _cma(self, func, x0, sigma, budget, deadline):
        d = self.dim
        n = max(1, int(np.floor(4 + 3 * np.log(d))))          # 13 for d=5
        sigma = float(sigma)
        c1 = (2 + 2.0 / (d + 1.3)) / ((d + 2) ** 2 + n)
        cmu = min(0.5 - c1,
                  (2 + 2.0 / (d + 1.3) - (d + n + 2) / ((d + 2) + n)) /
                  ((d + 2) ** 2 + n))
        cs = (n + 1.5 + 2.0 / (d + n)) / (d + n + 5)
        c_sigma = (1 + 2 * max(0.0, np.sqrt((n - 1) / (d + 1)) - 1)) * cs
        damps = (1 + 2 * max(0.0, np.sqrt((n - 1) / (d + 1)) - 1) + cs)
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        pc = np.zeros(d)
        ps = np.zeros(d)
        B = np.eye(d)
        D = np.ones(d)
        C = np.eye(d)

        def C_matvec(v):
            tmp = D * (B.T @ v)
            return D * (B @ tmp)

        x = np.asarray(x0, dtype=float).copy()
        fx = float(func(x))
        best_x = x.copy()
        best_f = fx
        it = 0
        while it < budget:
            if time.time() > deadline:
                break
            # sample
            z = rng.standard_normal((n, d))
            Y = z @ (B * D)
            pop = x[None, :] + sigma * Y
            vals = np.empty(n)
            for i in range(n):
                vals[i] = float(func(pop[i]))
            it += n
            order = np.argsort(vals)
            pop = pop[order]
            vals = vals[order]
            w = np.log(np.arange(1, n + 1)) - np.log(n)
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w[:n // 2] ** 2)

            m = pop[:n // 2].T @ w[:n // 2]
            pc = (1 - cs) * pc + cs * np.sqrt(mu_eff) * (m - x) / sigma
            norm_pc = np.linalg.norm(pc) / (
                np.sqrt(1 - (1 - cs) ** (2 * (it + 1) / (d * cs))) * chiN)
            hs = 1.0 if norm_pc < (1.4 + 2.0 / (d + 1)) * (1 - (1 - cs) ** (it + 1)) else 0.0
            ps = (1 - c_sigma) * ps + hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - x) / sigma
            delta_hs = (1 - c_sigma) ** (-0.5) - 1
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) / chiN - 1) * delta_hs
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(max(sigma, 1e-12), 20.0)

            rank_mu = np.outer(m, m) / mu_eff
            rank_one = np.outer(pc, pc)
            C = (1 + c1 * hsig + 2 * cmu - c1) * C + c1 * (hsig * (rank_one - C) - rank_one) + cmu * rank_mu
            # symmetric eigen-decomposition of C
            C = 0.5 * (C + C.T)
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-18))
            B = B2
            D = D2
            x = m
            fx = float(vals[0])
            if fx < best_f:
                best_f = fx
                best_x = m.copy()
        return best_x, best_f

    # ------------------------------------------------------------------ #
    # Driver
    # ------------------------------------------------------------------ #
    def __call__(self, func):
        global rng
        rng = np.random.default_rng(0)
        t0 = time.time()
        # Per-instance budget: 3 instances share a 330 s kill window; keep a
        # margin so the last instance still finishes.
        per_instance = min(100.0, (self.budget - 1) / 25.0)
        deadline = t0 + per_instance

        # diverse multi-start centers
        centers = [np.zeros(self.dim)]
        for _ in range(5):
            centers.append(rng.uniform(-4.5, 4.5, self.dim))

        best_x = centers[0].copy()
        best_f = float("inf")
        for c in centers:
            if time.time() > deadline - 1.0:
                break
            x0 = np.clip(c, -5.0, 5.0)
            bx, bf = self._cma(func, x0, 2.0, self.budget, deadline)
            if bf < best_f:
                best_f = bf
                best_x = bx

        # final safety: make sure we have at least one valid call
        if np.isinf(best_f):
            best_x = np.zeros(self.dim)
            best_f = float(func(best_x))
        return best_x, best_f
# EVOLVE-BLOCK-END
