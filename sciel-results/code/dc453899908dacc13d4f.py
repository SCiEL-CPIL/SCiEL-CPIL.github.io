"""Rewritten optimizer: global sampling + CMA-ES for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Global-to-local optimizer: stratified/random sampling then CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        t_start = time.time()
        DEADLINE = 300.0          # hard wall-clock limit (evaluator kills at 330s)
        SAFE = 3.0                # safety margin

        # ---- Phase 1: global exploration over the known minimum box [-4, 4]^dim ----
        n_global = max(5, min(75, budget // 6))
        n_local = budget - n_global

        best_x = None
        best_f = np.inf
        calls = 0

        def time_ok():
            return (time.time() - t_start) < (DEADLINE - SAFE)

        # stratified grid over [-4, 4]^dim (2 points per axis)
        grid = []
        for i in range(2 ** dim):
            pts = []
            for d in range(dim):
                bit = (i >> d) & 1
                pts.append(-4.0 + bit * 8.0)
            grid.append(pts)
        grid = np.array(grid)
        for p in grid:
            if not time_ok() or calls >= n_global:
                break
            fx = func(p)
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, p.copy()

        # random fill to reach n_global
        while calls < n_global and time_ok():
            p = rng.uniform(-4.0, 4.0, dim)
            fx = func(p)
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, p.copy()

        # ---- Phase 2: CMA-ES local optimization ----
        x0 = np.asarray(best_x, dtype=float)
        popsize = 4 * (dim + 2)          # 24 for dim=5
        c_sigma = 1.0 / (dim + 4)
        c_c = (4 + dim / 4) / (dim + 4 + 4)
        c_mu1 = (popsize // 2) ** 2 / popsize ** 2
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = (2 * (c_mu1 * (dim + 0.9) - c1)) / ((1 + c1) * (dim ** 2 + 1.7))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        damps = 1 + 2 * max(0, np.sqrt((popsize / 2 - 1) / (dim + 1)) - 1) + c_sigma
        damps_c = 1 + 2 * max(0, np.sqrt((popsize / 2 - 1) / (dim + 1)) - 1) + c_c
        w = np.log(popsize / 2) - np.log(np.arange(1, popsize + 1))
        w /= w.sum()
        mu_eff = 1 / (w @ w)

        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        inv_sqrt_C = np.eye(dim)

        def inv_sqrt(mat):
            val, vec = np.linalg.eigh(mat)
            val = np.maximum(val, 1e-20)
            return vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T

        for _ in range(n_local):
            if not time_ok() or calls >= budget:
                break
            # sample population
            samples = x0 + sigma * (inv_sqrt_C @ rng.standard_normal((popsize, dim)).T).T
            vals = np.empty(popsize)
            for i in range(popsize):
                if not time_ok():
                    vals[i] = np.inf
                    continue
                vals[i] = func(np.clip(samples[i], -5.0, 5.0))
                calls += 1
            order = np.argsort(vals, kind="stable")
            top = order[:popsize // 2]
            y_w = samples[top]
            mean_new = y_w.T @ w
            diff = mean_new - x0
            # update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (diff / sigma))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls))) < (1.4 + 2 / (dim + 1)) * chi_n:
                pc = (1 - c_c) * pc + c_c * (diff / sigma)
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * sum(wi * np.outer(yi - x0, yi - x0) for wi, yi in zip(w, y_w))
            x0 = mean_new
            sigma *= np.exp((c_sigma / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)
            inv_sqrt_C = inv_sqrt(C)
            # track best
            m = vals[order[0]]
            if m < best_f:
                best_f = m
                best_x = np.clip(samples[order[0]], -5.0, 5.0).copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
