"""Optimizer for BBOB-style Katsuura (f23) with time-aware hybrid search."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s kill
        self._calls_made = 0

        # Global best tracking
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        # ---- Phase 1: Multi-start (1+1)-ES for global exploration ----
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        n_starts = 30
        starts = []
        starts.append(np.zeros(dim))
        for i in range(dim):
            for v in [-4.0, 4.0]:
                p = np.zeros(dim)
                p[i] = v
                starts.append(p)
        while len(starts) < n_starts:
            p = rng.uniform(-4.5, 4.5, dim)
            starts.append(p)

        # Sort starts by norm (origin first)
        starts_sorted = sorted(starts, key=lambda p: np.linalg.norm(p))

        # Allocate budget: top 10 get more, rest get less
        remaining = budget
        top_n = min(10, n_starts)
        per_top = max(5, int(remaining * 0.55 / top_n))
        per_rest = max(3, int(remaining * 0.45 / max(1, n_starts - top_n)))

        for idx, start in enumerate(starts_sorted):
            if time.time() > deadline - 30:
                break
            if self._calls_made >= budget:
                break

            x = np.clip(start, -5, 5).astype(float)
            fx = func(x)
            self._calls_made += 1
            update_best(x, fx)

            steps = per_top if idx < top_n else per_rest
            sigma = 1.5
            for s in range(steps):
                if time.time() > deadline - 10:
                    break
                if self._calls_made >= budget:
                    break
                self._calls_made += 1
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma = min(sigma * 1.3, 3.0)
                else:
                    sigma = max(sigma * 0.75, 0.01)

        # ---- Phase 2: CMA-ES local refinement around best point ----
        if self._calls_made < budget and best_x is not None:
            cma_budget = budget - self._calls_made
            if cma_budget > 20:
                self._cma_es(func, best_x.copy(), best_f, cma_budget, deadline)

        # ---- Phase 3: Fine-tuning with very small (1+1)-ES ----
        if self._calls_made < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            for _ in range(budget - self._calls_made):
                if time.time() > deadline - 2:
                    break
                if self._calls_made >= budget:
                    break
                self._calls_made += 1
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma = min(sigma * 1.2, 0.5)
                else:
                    sigma = max(sigma * 0.8, 0.001)

        return best_x, best_f

    def _cma_es(self, func, x0, f0, budget, deadline):
        """Simple CMA-ES for local refinement."""
        dim = self.dim
        rng = np.random.default_rng(123)

        x = x0.copy()
        f = f0
        sigma = 0.5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + dim / 3) / (dim + 4 + 2)
        cs = (dim + 1.7) / (dim + 9 + 2 * cc)
        cmu = min(0.3, (2 * (dim - 1.7) + 1) / ((dim + 2) * (dim + 2)))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        lam = max(4, int(4 + 3 * np.log(dim)))
        w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        w = w / w.sum()
        mueff = 1 / w.sum()
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        calls = 0
        best_local_f = f

        for gen in range(budget // lam + 1):
            if time.time() > deadline - 5:
                break
            if calls >= budget:
                break

            offspring = []
            offspring_f = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline - 5:
                    break
                z = rng.standard_normal(dim)
                # Use Cholesky of C for proper sampling
                try:
                    L = np.linalg.cholesky(C)
                    y = x + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                self._calls_made += 1
                offspring.append(y)
                offspring_f.append(fy)

            if not offspring:
                break

            order = np.argsort(offspring_f)
            offspring = [offspring[i] for i in order]
            offspring_f = [offspring_f[i] for i in order]

            if offspring_f[0] < best_local_f:
                best_local_f = offspring_f[0]
                if offspring_f[0] < f:
                    x = offspring[0].copy()
                    f = offspring_f[0]

            mu = min(int(mueff), lam // 2)
            delta = np.zeros(dim)
            for i in range(mu):
                delta += w[i] * (offspring[i] - x)
            x = x + delta
            x = np.clip(x, -5, 5)

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / max(sigma, 1e-10) * delta
            hsig = (np.linalg.norm(ps) / max(np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))), 1e-10)) / chi < 1.4 + 2 / (dim + 1)
            pc = (1 - cc) * pc + hsig * cc * delta
            C = (1 + cmu * cs * damps * (np.outer(ps, ps) - C) + cmu * (np.outer(delta, delta) - C)) * (1 - cmu)
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-10):
                C += 1e-8 * np.eye(dim)

            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-6, 5.0)

        return x, f
