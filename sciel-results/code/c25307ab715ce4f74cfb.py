"""Rewritten optimizer: multi-restart CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial sigma. Returns (best_x, best_fx, calls_used)."""
        n = self.dim
        calls = 0
        best_x = x0.copy()
        best_fx = func(x0)
        calls += 1

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        cmu = min(1 - 2 / (n + 1.3) ** 2,
                  2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        # Initialize
        x_mean = x0.copy().astype(float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        while calls < budget:
            if time.time() > deadline:
                break

            # Sample
            y = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                y[i] = x_mean + sigma * (C @ z)
                y[i] = np.clip(y[i], -5, 5)
                fy = func(y[i])
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort([func(yi) for yi in y])  # This would re-evaluate; need to store
            # Actually we need to store fitnesses. Let me restructure.
            pass

        return best_x, best_fx, calls

    def _cma_es_v2(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial sigma. Returns (best_x, best_fx, calls_used)."""
        n = self.dim
        calls = 0
        best_x = x0.copy()
        best_fx = func(x0)
        calls += 1

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        cmu = min(1 - 2 / (n + 1.3) ** 2,
                  2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        # Initialize
        x_mean = x0.copy().astype(float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        while calls < budget:
            if time.time() > deadline:
                break

            # Sample and evaluate
            y = np.zeros((lam, n))
            fy = np.zeros(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                y[i] = x_mean + sigma * (C @ z)
                y[i] = np.clip(y[i], -5, 5)
                fy[i] = func(y[i])
                calls += 1

            if calls >= budget:
                # Update best if any new point is better
                for i in range(lam):
                    if fy[i] < best_fx:
                        best_fx = fy[i]
                        best_x = y[i].copy()
                break

            # Sort by fitness
            order = np.argsort(fy)
            y_sorted = y[order]
            fy_sorted = fy[order]

            # Update best
            if fy_sorted[0] < best_fx:
                best_fx = fy_sorted[0]
                best_x = y_sorted[0].copy()

            # Weighted mean of top mu
            y_w = np.sum(w[:, None] * y_sorted[:mu], axis=0)
            x_mean_new = x_mean + (y_w - x_mean)

            # Cumulation
            pc = (1 - cc) * pc + cc * np.sqrt(cc * (2 - cc) * mueff) / np.linalg.norm(C) * (x_mean_new - x_mean)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.inv(np.sqrt(C)) @ (x_mean_new - x_mean) / sigma

            # Update covariance
            delta_hs = (np.linalg.norm(ps) / (1 - (1 - cs) ** (calls / (cs * n)))) > (1.4 + 2 / (n + 1)) * chi_n
            C = ((1 - cmu) * C +
                 cmu * (np.outer(pc, pc) +
                        np.sum(w[:, None, None] * np.outer(y_sorted[:mu] - x_mean, y_sorted[:mu] - x_mean), axis=0) / sigma ** 2 +
                        (delta_hs * (1 + cmu) - 1) * np.outer(ps, ps)))

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = (eigvecs * eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            x_mean = x_mean_new

            # Update sigma
            sigma *= np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # 5s safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-restart strategy
        # First restart: full CMA-ES with larger budget
        # Subsequent restarts: shorter CMA-ES with different initializations

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Phase 1: Initial exploration with CMA-ES
        x0 = rng.uniform(-5, 5, dim)
        sigma0 = 2.0
        budget_phase1 = int(budget * 0.6)

        if time.time() < deadline:
            bx, bfx, calls_used = self._cma_es_v2(func, x0, sigma0, budget_phase1, rng, deadline)
            total_calls += calls_used
            if bfx < best_fx:
                best_fx = bfx
                best_x = bx.copy()

        # Phase 2: Local refinement around best found
        remaining = budget - total_calls
        if remaining > 10 and best_x is not None and time.time() < deadline:
            # Use smaller sigma for local search
            sigma_local = 0.5
            bx, bfx, calls_used = self._cma_es_v2(func, best_x, sigma_local, remaining, rng, deadline)
            total_calls += calls_used
            if bfx < best_fx:
                best_fx = bfx
                best_x = bx.copy()

        # Phase 3: A few more restarts if time allows
        remaining = budget - total_calls
        restarts_done = 1
        while remaining > 20 and time.time() < deadline and restarts_done < 3:
            x0 = best_x + 0.5 * rng.standard_normal(dim)
            x0 = np.clip(x0, -5, 5)
            sigma0 = 0.3
            bx, bfx, calls_used = self._cma_es_v2(func, x0, sigma0, remaining, rng, deadline)
            total_calls += calls_used
            if bfx < best_fx:
                best_fx = bfx
                best_x = bx.copy()
            remaining = budget - total_calls
            restarts_done += 1

        # Final local polish with very small steps
        remaining = budget - total_calls
        if remaining > 5 and best_x is not None and time.time() < deadline:
            for _ in range(min(remaining, 20)):
                if time.time() > deadline:
                    break
                dx = 0.05 * rng.standard_normal(dim)
                x_new = np.clip(best_x + dx, -5, 5)
                fx_new = func(x_new)
                total_calls += 1
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
