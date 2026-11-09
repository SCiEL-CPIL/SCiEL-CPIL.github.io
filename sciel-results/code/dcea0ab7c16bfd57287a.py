"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin).

Combines:
  - Broad multi-start (1+1)-ES exploration (inspiration 1) to escape the
    Lunacek Rastrigin ridges and locate both basins.
  - CMA-ES covariance adaptation (current program) for directed refinement.
  - A stochastic hill-climb with geometric step decay + random-perturbation
    restarts as a fast memoryless polish (new structural direction).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        calls = 0

        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                return 1e10
            if not np.isfinite(val):
                return 1e10
            return val

        best_x = None
        best_f = np.inf

        # ---------------- Phase 1: broad multi-start (1+1)-ES -------------
        # Lunacek has two basins; sample origin + diverse random points.
        n_starts = 6
        start_points = [np.zeros(n)]
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4.0, 4.0, n))

        # ~40% of budget spread across the exploratory (1+1)-ES runs.
        per_start = max(15, (budget * 40) // (100 * n_starts))

        for sp in start_points:
            if time.time() > deadline or calls >= budget - 20:
                break
            x, fx = self._es_restart(safe_func, sp, n, per_start, deadline, rng,
                                     initial_step=1.5)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_func(best_x)

        # ---------------- Phase 2: CMA-ES refinement ----------------------
        if calls < budget - 15 and time.time() < deadline:
            remaining = budget - calls
            cma_budget = min(remaining, 300)
            x2, f2 = self._cma_es(safe_func, best_x.copy(), n, cma_budget,
                                  deadline, rng)
            if f2 < best_f:
                best_f = f2
                best_x = x2.copy()

        # ---------------- Phase 3: stochastic hill-climb polish -----------
        if calls < budget - 5 and time.time() < deadline:
            remaining = budget - calls
            x3, f3 = self._hill_climb(safe_func, best_x.copy(), n, remaining,
                                      deadline, rng)
            if f3 < best_f:
                best_f = f3
                best_x = x3.copy()

        return best_x, best_f

    def _es_restart(self, func, x0, dim, max_calls, deadline, rng,
                    initial_step=1.5):
        """(1+1)-ES with 1/5 success-rule adaptive step size."""
        x = x0.copy()
        fx = func(x)
        sigma = initial_step
        c_sigma = 0.5
        d_sigma = 1.0 + 0.35 * min(dim / 3.0, 1.0)

        for _ in range(max_calls):
            if time.time() > deadline:
                break
            z = rng.standard_normal(dim)
            x_new = x + sigma * z
            x_new = np.clip(x_new, -5.0, 5.0)
            fx_new = func(x_new)
            if fx_new < fx:
                sigma *= np.exp(c_sigma / d_sigma)
                x = x_new
                fx = fx_new
            else:
                sigma *= np.exp(-c_sigma / d_sigma)
            sigma = np.clip(sigma, 1e-4, 5.0)
            if sigma < 1e-5:
                break
        return x, fx

    def _hill_climb(self, func, x0, dim, max_calls, deadline, rng):
        """Stochastic hill-climb: geometric step decay + random-perturbation
        restarts (memoryless single-point updates with explicit diversity)."""
        x = x0.copy()
        fx = func(x)
        sigma = 0.5
        decay = 0.985
        stall = 0
        stall_limit = 30

        for _ in range(max_calls):
            if time.time() > deadline:
                break
            z = rng.standard_normal(dim)
            x_new = x + sigma * z
            x_new = np.clip(x_new, -5.0, 5.0)
            fx_new = func(x_new)
            if fx_new < fx:
                x = x_new
                fx = fx_new
                stall = 0
            else:
                stall += 1
            sigma *= decay
            sigma = np.clip(sigma, 1e-5, 5.0)
            # Random-perturbation restart to re-inject diversity on stagnation.
            if stall >= stall_limit:
                x = x + rng.uniform(-1.0, 1.0, dim)
                x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                sigma = 0.5
                stall = 0
            if sigma < 1e-5:
                break
        return x, fx

    def _cma_es(self, func, x0, dim, budget, deadline, rng):
        """CMA-ES with covariance adaptation for directed refinement."""
        pop_size = 4 + int(3 * np.log(dim))
        pop_size = max(pop_size, 4)

        c_sigma = (1 + 2 * max(0, np.sqrt((pop_size - 1) / dim) - 1)) / (dim + 2)
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - dim - 2) /
                  ((dim + 2) ** 2 + pop_size))
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x_mean = x0.copy()
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = func(x_mean)

        for gen in range(budget // pop_size + 1):
            if time.time() > deadline:
                break

            offspring, offspring_f = [], []
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x_new = np.clip(x_mean + sigma * y, -5.0, 5.0)
                f_new = func(x_new)
                offspring.append(x_new)
                offspring_f.append(f_new)

            if len(offspring) < 2:
                break

            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]

            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            x_new_mean = np.sum([weights[i] * offspring[i] for i in range(mu)],
                                axis=0)

            diff = [offspring[i] - x_mean for i in range(mu)]
            C = ((1 - c_1 - c_mu) * C
                 + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * C)
                 + c_mu * np.sum([np.outer(diff[i], diff[i]) for i in range(mu)],
                                 axis=0))

            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except Exception:
                C = np.eye(dim)

            ps = ((1 - c_sigma) * ps
                  + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma
                  * (B @ (D ** -1)) @ (x_new_mean - x_mean))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            pc = ((1 - c_c) * pc
                  + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma
                  * (x_new_mean - x_mean))

            x_mean = np.clip(x_new_mean, -5.0, 5.0)

            if np.linalg.norm(pc) > (1.4 + 1 / c_c) * chi_n:
                break
            if sigma < 1e-6:
                break
            if np.max(C) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
