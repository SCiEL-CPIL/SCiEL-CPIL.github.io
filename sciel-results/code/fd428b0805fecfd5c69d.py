"""Optimizer for BBOB f9 (rotated Rosenbrock) on [-5,5]^5."""
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES with an explicit wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # --- CMA-ES core ------------------------------------------------------
    def _cma(self, func, x0, sigma0, budget, rng, deadline):
        n = self.dim
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (n + 4 + mueff)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff * c_sigma + c_c) / ((n + 1.3) ** 2 + mueff)
        c_mu = (2 * max(0.0, mueff * c_sigma - c_c - 1) / (n + 2)
                - (2 - mueff * c_sigma + c_c)) / ((n + 1.3) ** 2 - mueff)
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        x = np.clip(np.array(x0, dtype=float), -5, 5)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        age = 0

        fx = func(x)
        calls = 1
        best_x, best_f = x, fx

        while calls + lam <= budget:
            if time.time() > deadline:
                break
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-12, None)
            D = np.sqrt(eigval)
            B = eigvec
            z = rng.standard_normal((lam, n))
            pop = np.array([x + sigma * (B @ (D * zi)) for zi in z])
            pop = np.clip(pop, -5, 5)
            fvals = np.array([func(xi) for xi in pop])
            calls += lam
            order = np.argsort(fvals)
            if fvals[order[0]] < best_f:
                best_f = fvals[order[0]]
                best_x = pop[order[0]].copy()
            yw = np.array([pop[i] for i in order[:mu]])
            m = np.sum(w[:, None] * yw, axis=0)

            ps = (1 - c_sigma) * ps + np.sqrt(mueff * c_sigma * (2 - c_sigma)) * (B.T @ (D * (m - x) / sigma))
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(max(sigma, 1e-12), 1e6)
            age += 1
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * age))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(mueff * c_c * (2 - c_c)) * (m - x)

            C = (1 - c_1 - c_mu) * C
            if hsig:
                C += c_1 * (np.outer(pc, pc) + c_c * (2 - c_c) * np.eye(n))
            else:
                C += c_1 * np.outer(pc, pc)
            C += c_mu * np.sum(w[:, None, None] * np.outer(yw - x, yw - x), axis=0)
            C = (C + C.T) / 2.0

            x = m
            fx = fvals[order[0]]
        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 325.0  # safety margin before 330 s kill

        schedule = [
            (0.55, 0, 1.5),
            (0.20, 1, 1.0),
            (0.15, 2, 0.6),
            (0.10, 3, 0.4),
        ]

        best_x = None
        best_f = np.inf
        used = 0
        for frac, seed, sigma0 in schedule:
            if time.time() > deadline:
                break
            b = max(2 * (dim + 1) + 4, int(frac * budget))
            b = min(b, budget - used)
            if b < 2 * (dim + 1) + 4:
                break
            rng = np.random.default_rng(seed)
            x0 = rng.uniform(-5, 5, dim)
            x, f = self._cma(func, x0, sigma0, b, rng, deadline)
            used += b
            if f < best_f:
                best_f, best_x = f, x
            if time.time() > deadline:
                break

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
