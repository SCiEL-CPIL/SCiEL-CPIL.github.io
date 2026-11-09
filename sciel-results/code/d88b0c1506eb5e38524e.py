"""Optimizer for BBOB f21 (Gallagher 101 peaks) via CMA-ES."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with a global scan to seed the mean and a deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 295.0  # safety margin below 300 s

        rng = np.random.default_rng(12345)
        N = 10  # population size

        # ---- global scan to seed the mean ----
        scan_n = 50
        xs = rng.uniform(-5.0, 5.0, (scan_n, dim))
        vals = np.array([func(x) for x in xs])
        bi = int(np.argmin(vals))
        x = xs[bi].copy()
        fx = vals[bi]
        fmin = fx
        calls = scan_n

        # ---- CMA-ES state ----
        N = 10
        mu = N // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 0.5) / (dim + 2) - 1))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        d_sigma = c_sigma * (1 + c_mu / c_1) + 1
        hsig = (1.34 + 2 / (dim + 1)) * chiN

        sigma = 1.0
        C = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def sample():
            z = rng.standard_normal((N, dim))
            return x + sigma * (z @ Csqrt.T)

        def inv_sqrtm(M):
            eigval, eigvec = np.linalg.eigh(M)
            eigval = np.clip(eigval, 1e-12, None)
            return eigvec @ (np.diag(eigval ** -0.5) @ eigvec.T)

        while calls < self.budget:
            if time.time() > deadline:
                break

            Y = sample()
            F = np.array([func(y) for y in Y])
            calls += N

            order = np.argsort(F)
            Ys = Y[order]
            Fs = F[order]

            for k in range(mu):
                if Fs[k] < fmin:
                    fmin = Fs[k]
                    x = Ys[k].copy()

            ys = (Ys[:mu] - x) / sigma
            yw = w[:, None] * ys

            # step size adaptation
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (Cinv @ (w @ ys))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - scan_n + 1))) < hsig:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))

            # covariance adaptation
            pc = (1 - c_c) * pc + c_c * (F[order[0]] < F[order[-1]]) * (yw[:, 0])
            v = pc + (c_1 / c_c - 1) * x
            C = (1 + c_1 * (c_mu * np.sum(w ** 2) - 2) + c_mu) * C
            C += c_mu * (yw.T @ yw) - c_1 * (pc @ pc + v @ v)

            # recompute matrix square root and inverse
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-12, None)
            Csqrt = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T
            Cinv = inv_sqrtm(C)

            # keep sigma sane
            sigma = min(max(sigma, 1e-12), 10.0)

            if time.time() > deadline:
                break

        return x, fmin
# EVOLVE-BLOCK-END
