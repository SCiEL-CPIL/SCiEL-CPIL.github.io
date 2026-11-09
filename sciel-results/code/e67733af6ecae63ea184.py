"""Derivative-free optimizer for the BBOB Discus (f11) landscape."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize as _minimize


class _CMAES:
    """Minimal CMA-ES with sigma adaptation (1/(1-c) rule) and boundary handling."""

    def __init__(self, x0, f0, dim, sigma0, bounds, rng, maxit, time_limit):
        self.dim = dim
        self.bound_lo = np.asarray(bounds[0], dtype=float)
        self.bound_hi = np.asarray(bounds[1], dtype=float)
        self.rng = rng
        self.maxit = maxit
        self.time_limit = time_limit
        self.t0 = time.time()

        n = dim
        self.mu = max(1, n // 2)
        self.w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = self.w / self.w.sum()
        self.mueff = 1.0 / self.w.sum()
        self.c_sigma = (self.mueff + 2) / (n + 4)
        self.d_sigma = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.c_1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        self.x = np.clip(np.asarray(x0, dtype=float), self.bound_lo, self.bound_hi)
        self.f = float(f0)
        self.sigma = float(sigma0)
        self.C = np.eye(n)
        self.Csqrt = np.eye(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self._chisq = np.zeros(n)
        self._chisq_sum = float(n)
        self._chisq_count = 0

    def _mahalanobis_sq(self, v):
        return float(np.dot(v, np.linalg.solve(self.C, v)))

    def _update_chisq(self, d):
        self._chisq = 0.98 * self._chisq + 0.02 * d
        self._chisq_sum += 0.02 * d
        self._chisq_count += 1

    def _adapt_sigma(self, d, ds):
        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (d - self.chi_n))
        if self.sigma > 1e3:
            self.sigma = 1e3
        if self.sigma < 1e-12:
            self.sigma = 1e-12

    def _step(self, ds, d, z):
        self._update_chisq(d)
        self._adapt_sigma(d, ds)
        self.pc = (1 - self.c_c) * self.pc + math.sqrt(self.c_c * (2 - self.c_c) * self.mueff) * self.sigma * self.Csqrt @ z
        hsig = (np.linalg.norm(self.pc) / math.sqrt(1 - (1 - self.c_c) ** (self._chisq_count + 1)) < (1 + self.c_c) * self.chi_n) and 1.0 or 0.0
        self.ps = (1 - self.c_sigma) * self.ps + math.sqrt(self.c_sigma * (2 - self.c_sigma) * hsig) * z
        delta = self.c_1 * ((1 + (1 - hsig) * self.mueff / 2) * np.outer(self.pc, self.pc) - np.eye(self.dim))
        delta += self.c_mu * sum(self.w[i] * np.outer(self._chisq[i], self._chisq[i]) for i in range(self.mu))
        self.C = (1 + self.c_1 + self.c_mu - self.c_1 / self.dim) * self.C + delta
        self.C = 0.5 * (self.C + self.C.T)
        self.Csqrt = np.linalg.cholesky(self.C + 1e-12 * np.eye(self.dim))

    def run(self, func):
        n = self.dim
        best_x, best_f = self.x.copy(), self.f
        for _ in range(self.maxit):
            if time.time() - self.t0 > self.time_limit:
                break
            # generate population
            pop = []
            for i in range(self.mu):
                z = self.rng.standard_normal(n)
                y = self.x + self.sigma * (self.Csqrt @ z)
                y = np.clip(y, self.bound_lo, self.bound_hi)
                pop.append((y, func(y), z))
            pop.sort(key=lambda t: t[1])

            # weighted mean
            m = np.zeros(n)
            for i in range(self.mu):
                m += self.w[i] * pop[i][0]
            # step direction
            z = np.zeros(n)
            for i in range(self.mu):
                z += self.w[i] * pop[i][2]
            d = float(np.linalg.norm(z))
            ds = d / self.chi_n
            self._step(ds, d, z)
            self.x = m
            self.f = float(pop[0][1])
            if self.f < best_f:
                best_f = self.f
                best_x = m.copy()
            # stop if sigma tiny or step too small
            if self.sigma < 1e-12 or ds < 1e-12:
                break
        return best_x, best_f


def _latin_hypercube(rng, dim, n):
    pts = np.empty((n, dim))
    for j in range(dim):
        perm = rng.permutation(n)
        pts[:, j] = (perm + rng.random(n)) / n
    return pts


def _halton(dim, n, skip=1000):
    pts = np.empty((n, dim))
    for j in range(dim):
        base = j + 2
        col = np.empty(n)
        for i in range(n):
            k = i + skip
            f, r = 1.0, 0.0
            while k > 0:
                k, rem = divmod(k, base)
                f /= base
                r += f * rem
            col[i] = r
        pts[:, j] = col
    return pts


def _nelder_mead(func, x0, bounds, rng, maxit, time_limit):
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    dim = len(x0)
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)

    def wrapped(v):
        v = np.clip(v, lo, hi)
        return func(v)

    simplex = [x0.copy()]
    for j in range(dim):
        p = x0.copy()
        step = 0.5 if x0[j] + 0.5 <= hi[j] else -0.5
        p[j] += step
        simplex.append(p)
    res = _minimize(wrapped, x0, method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': maxit,
                             'adaptive': True, 'initial_simplex': np.array(simplex)})
    if res.success or (res.fun is not None and np.isfinite(res.fun)):
        return np.clip(res.x, lo, hi), float(res.fun)
    return x0, func(x0)


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin under 300 s
        rng = np.random.default_rng(12345)
        bounds = (-5.0, 5.0)

        # ---- Phase 1: global multi-start sampling (fixed small fraction) ----
        n_global = max(20, min(80, budget // 10))
        candidates = []
        # random
        for _ in range(max(1, n_global // 3)):
            candidates.append(rng.uniform(-5, 5, dim))
        # Latin hypercube
        candidates.extend(_latin_hypercube(rng, dim, max(1, n_global // 3)))
        # Halton
        candidates.extend(_halton(dim, max(1, n_global // 3)))
        # center + corners
        candidates.append(np.zeros(dim))
        for _ in range(min(8, n_global // 4)):
            candidates.append(rng.choice([-5.0, 5.0], size=dim))

        best_x, best_f = None, np.inf
        for c in candidates:
            c = np.clip(c, -5, 5)
            f = func(c)
            if f < best_f:
                best_f = f
                best_x = c.copy()

        # ---- Phase 2: fast Nelder-Mead polish of the best few seeds ----
        # keep the best few distinct seeds from global phase
        seeds = []
        seen = []
        for c in candidates:
            key = tuple(np.round(np.clip(c, -5, 5), 3))
            if key not in seen:
                seen.append(key)
                seeds.append(c)
        # polish top 4
        for c in seeds[:4]:
            if time.time() > deadline - 30:
                break
            px, pf = _nelder_mead(func, c, bounds, rng, maxit=60, time_limit=deadline - time.time() - 5)
            if pf < best_f:
                best_f = pf
                best_x = px.copy()

        # ---- Phase 3: CMA-ES from the best point, with remaining budget ----
        remaining = budget - (len(candidates) + 4 * 60)
        if remaining > 30 and time.time() < deadline - 60:
            sigma0 = 1.0
            cma = _CMAES(best_x, best_f, dim, sigma0, bounds, rng,
                         maxit=max(20, remaining // 10),
                         time_limit=deadline - time.time() - 10)
            cx, cf = cma.run(func)
            if cf < best_f:
                best_f = cf
                best_x = cx.copy()

        # ---- Phase 4: final local polish of the global best ----
        if time.time() < deadline - 5:
            px, pf = _nelder_mead(func, best_x, bounds, rng, maxit=80,
                                  time_limit=deadline - time.time() - 2)
            if pf < best_f:
                best_f = pf
                best_x = px.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
