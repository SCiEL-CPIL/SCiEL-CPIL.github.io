"""Rewritten optimizer: parallel adaptive strategies with shared budget + deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class _NelderMead:
    def __init__(self, func, x0, dim, lo, hi, rng, deadline, shared):
        self.func = func
        self.dim = dim
        self.lo = lo
        self.hi = hi
        self.rng = rng
        self.deadline = deadline
        self.shared = shared
        self.x = np.array(x0, dtype=float)
        self.f = func(self.x)
        self._build_simplex()

    def _build_simplex(self):
        self.simplex = np.empty((self.dim + 1, self.dim))
        self.simplex[0] = self.x
        for i in range(self.dim):
            p = self.x.copy()
            p[i] += 0.25
            p[i] = min(max(p[i], self.lo), self.hi)
            self.simplex[i + 1] = p
        self.fvals = np.empty(self.dim + 1)
        self.fvals[0] = self.f
        for i in range(1, self.dim + 1):
            self.fvals[i] = self.func(self.simplex[i])

    def _cull(self, k):
        self.simplex = np.delete(self.simplex, k, axis=0)
        self.fvals = np.delete(self.fvals, k)

    def _add(self, p, f):
        self.simplex = np.vstack([self.simplex, p])
        self.fvals = np.append(self.fvals, f)

    def _reflect(self, p, f):
        self.simplex[-1] = p
        self.fvals[-1] = f

    def _cull_add(self, p, f):
        self._cull(self.dim)
        self._add(p, f)

    def _reflect_expand(self, p, f):
        self._cull_add(p, f)
        self._add(self.x, self.f)

    def step(self):
        order = np.argsort(self.fvals)
        self.simplex = self.simplex[order]
        self.fvals = self.fvals[order]
        self.x = self.simplex[0]
        self.f = self.fvals[0]
        if self.f < self.shared.best_f:
            self.shared.best_f = self.f
            self.shared.best_x = self.x.copy()
        if self.dim + 1 != len(self.fvals):
            self._build_simplex()
            return
        worst = self.simplex[-1].copy()
        centroid = self.simplex[:-1].mean(axis=0)
        x_r = centroid + (centroid - worst)
        x_r = np.clip(x_r, self.lo, self.hi)
        f_r = self.func(x_r)
        if f_r < self.fvals[0]:
            x_e = centroid + 2.0 * (centroid - worst)
            x_e = np.clip(x_e, self.lo, self.hi)
            f_e = self.func(x_e)
            if f_e < f_r:
                self._cull_add(x_e, f_e)
            else:
                self._cull_add(x_r, f_r)
        elif f_r < self.fvals[-2]:
            self._reflect(x_r, f_r)
        else:
            x_c = centroid + 0.5 * (worst - centroid)
            x_c = np.clip(x_c, self.lo, self.hi)
            f_c = self.func(x_c)
            if f_c < self.fvals[-1]:
                self._cull_add(x_c, f_c)
            else:
                self._shrink()

    def _shrink(self):
        x0 = self.simplex[0]
        for i in range(1, self.dim + 1):
            self.simplex[i] = x0 + 0.5 * (self.simplex[i] - x0)
            self.fvals[i] = self.func(self.simplex[i])


class _CMAES:
    def __init__(self, func, x0, dim, lo, hi, rng, deadline, shared):
        self.func = func
        self.dim = dim
        self.lo = lo
        self.hi = hi
        self.rng = rng
        self.deadline = deadline
        self.shared = shared
        self.mean = np.array(x0, dtype=float)
        self.f_mean = func(self.mean)
        self.sigma = 1.5
        self.C = np.eye(dim)
        self.Cinv = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.c_sigma = 1.0 / (dim + 4)
        self.c_c = (4 + dim / 4) / (dim + 4 + 3)
        self.d_sigma = 1 + 2 * max(0, (np.sqrt((dim + 1) / 3 - 1) - 1) / (dim + 3))
        self.chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim * dim))
        self.lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        self._chisqrt = np.sqrt(1 - np.exp(-2 / (dim + 1)))

    def _decomp(self):
        D, B = np.linalg.eigh(self.C)
        self.B = B
        self.D = np.sqrt(np.maximum(D, 1e-12))

    def _eval(self, x):
        return self.func(np.clip(x, self.lo, self.hi))

    def _step_c(self):
        self.pc = (1 - self.c_c) * self.pc + self.c_c * (self.ps / self.sigma)
        c1 = (2 / ((self.dim + 1.3) ** 2 + self.dim)) * (1 - self.c_c) + self.c_c
        cmu = min(1 - c1, 2 * (self.lam / 2 - 2 + 2 / self.dim) / ((self.dim + 2) ** 2 + self.lam))
        self.C = (1 - self.c_c * (c1 + 2 * cmu)) * self.C + self.c_c * np.outer(self.pc, self.pc)
        if cmu > 0:
            self.C += cmu * (self.v1 - self.v2)
        self._decomp()

    def _step_sigma(self):
        self.sigma *= np.exp(self.c_sigma * (np.linalg.norm(self.ps) / self.chi - 1))
        self.sigma = min(max(self.sigma, 1e-6), 20.0)

    def _reset(self):
        self.C = np.eye(self.dim)
        self.Cinv = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.sigma = 1.5
        self._decomp()

    def step(self):
        if self.dim + 1 != len(self.simplex) if hasattr(self, 'simplex') else True:
            pass
        if not hasattr(self, 'B'):
            self._decomp()
        w, mu_eff = self._weights()
        x = self.mean + self.sigma * self.B @ (self.D * self.rng.standard_normal((self.lam, self.dim)))
        fx = np.array([self._eval(xi) for xi in x])
        idx = np.argsort(fx)
        x = x[idx]
        fx = fx[idx]
        if fx[0] < self.shared.best_f:
            self.shared.best_f = fx[0]
            self.shared.best_x = x[0].copy()
        self.mean = x[:self.lam // 2 + 1] @ w[:self.lam // 2 + 1]
        y = (x[:self.lam // 2 + 1] - self.mean) / (self.sigma + 1e-12)
        self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * mu_eff) * (self.B.T @ (self.D * y.T)).T.mean(axis=0)
        self.v1 = np.einsum('i,ij,j->j', y.T.mean(axis=0), self.Cinv, y.T.mean(axis=0))
        self.v2 = np.einsum('ij,jk,ik->j', y, self.Cinv, y).mean(axis=1)
        self._step_c()
        self._step_sigma()
        if np.linalg.norm(self.ps) > 1.4 * self.chi + 2:
            self._reset()

    def _weights(self):
        mu = self.lam // 2 + 1
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = (w.sum() ** 2) / (w ** 2).sum()
        return w, mu_eff


class _Grid:
    def __init__(self, func, dim, lo, hi, rng, deadline, shared):
        self.func = func
        self.dim = dim
        self.lo = lo
        self.hi = hi
        self.rng = rng
        self.deadline = deadline
        self.shared = shared
        self.levels = []
        self._make_grid(2)
        self._make_grid(3)

    def _make_grid(self, n):
        pts = []
        for v in np.linspace(self.lo, self.hi, n):
            pts.append(v)
        grid = np.array(np.meshgrid(*([pts] * self.dim)), dtype=float).reshape(self.dim, -1).T
        self.levels.append(grid)

    def step(self):
        if not self.levels:
            return
        grid = self.levels.pop(0)
        for row in grid:
            if time.time() > self.deadline:
                return
            f = self.func(row)
            if f < self.shared.best_f:
                self.shared.best_f = f
                self.shared.best_x = row.copy()
        # refine around the current best
        if len(self.levels) < 2:
            self._make_grid(3)


class Optimizer:
    """Parallel adaptive strategies (Nelder-Mead, CMA-ES, grid) sharing a budget + deadline."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin under 300 s wall-clock
        dim = self.dim
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(0)

        class Shared:
            best_f = np.inf
            best_x = None

        shared = Shared()
        calls = [0]

        def counted(x):
            calls[0] += 1
            return func(np.clip(np.asarray(x, dtype=float), lo, hi))

        # ---- seed points: center, random, and a few coarse samples ----
        seed_pts = [np.zeros(dim), rng.uniform(lo, hi, dim)]
        for _ in range(4):
            seed_pts.append(rng.uniform(lo, hi, dim))

        strategies = []
        for sp in seed_pts:
            nm = _NelderMead(counted, sp, dim, lo, hi, rng, deadline, shared)
            if nm.f < shared.best_f:
                shared.best_f = nm.f
                shared.best_x = nm.x.copy()
            strategies.append(nm)

        # ---- CMA-ES from the best seed ----
        cma = _CMAES(counted, shared.best_x, dim, lo, hi, rng, deadline, shared)
        strategies.append(cma)

        # ---- coarse-to-fine grid for early exploration ----
        grid = _Grid(counted, dim, lo, hi, rng, deadline, shared)
        strategies.append(grid)

        # ---- main loop: round-robin until budget or deadline ----
        while calls[0] < self.budget and time.time() < deadline:
            for s in strategies:
                if calls[0] >= self.budget or time.time() >= deadline:
                    break
                if isinstance(s, _Grid) and not s.levels:
                    continue
                try:
                    s.step()
                except Exception:
                    pass
        return shared.best_x, shared.best_f
# EVOLVE-BLOCK-END
