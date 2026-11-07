"""Optimizer for BBOB f9 (rotated Rosenbrock) using CMA-ES."""
import time
import math
import numpy as np


class Optimizer:
    """CMA-ES continuous optimizer with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)

        # CMA-ES hyperparameters
        lam = 4 + int(math.floor(3.0 * math.log(n)))
        lam = max(lam, 2)
        mu = lam // 2
        w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        chiN = math.sqrt(n * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2)))
        cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c2 = 2.0 * (mu_eff - 2.0 + 2.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff)

        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        C = np.eye(n)
        D = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        gen = 0

        calls = 1
        while calls < budget:
            if time.time() > deadline - 5.0:
                break

            B = rng.standard_normal((lam, n))
            pop = x[None, :] + sigma * (B @ D.T)
            pop = np.clip(pop, -5.0, 5.0)
            fvals = np.empty(lam)
            for i in range(lam):
                fvals[i] = func(pop[i])
            calls += lam
            if calls >= budget:
                idx = int(np.argmin(fvals))
                if fvals[idx] < best_f:
                    best_f = fvals[idx]
                    best_x = pop[idx].copy()
                break

            order = np.argsort(fvals)
            sorted_f = fvals[order]
            sorted_pop = pop[order]

            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_pop[0].copy()

            m = np.zeros(n)
            for i in range(mu):
                m += w[i] * sorted_pop[i]

            Y = (sorted_pop[:mu] - x[None, :]) / sigma
            Dm = D @ ((m - x) / sigma)

            ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mu_eff) / chiN * Dm
            norm_ps = np.linalg.norm(ps)
            denom = math.sqrt(max(1.0 - (1.0 - cs) ** (2 * (gen + 1)), 1e-12))
            hsig = 1.0 if (norm_ps / denom) < (1.4 + 2.0 / (n + 1)) else 0.0

            pc = (1.0 - cc) * pc + hsig * cc * Dm

            C = (1.0 - c1 - c2 * hsig * (2.0 - hsig)) * C
            C += c1 * (np.outer(pc, pc) + (hsig - (1.0 - hsig)) * cc * (1.0 - cc) * C)
            for i in range(mu):
                C += c2 * w[i] * np.outer(Y[i], Y[i])

            x = m

            sigma *= math.exp((cs / chiN) * (norm_ps / chiN - 1.0))
            sigma = min(max(sigma, 1e-12), 1e6)

            C = 0.5 * (C + C.T)
            try:
                D = np.linalg.cholesky(C + 1e-9 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                D = np.eye(n)

            gen += 1

        return best_x, best_f
