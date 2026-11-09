"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES with multi-start strategy for Rosenbrock optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_run(self, func, x0, sigma0, max_evals, rng, deadline):
        """Run CMA-ES from x0 with initial step sigma0 for at most max_evals evaluations."""
        n = self.dim
        if n == 1:
            mu = 1
        else:
            mu = max(1, n // 2)
        lam = 2 * mu

        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = int(1 / np.sum(w**2))

        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mueff - 2) / (n + 2)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        x = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        best_x = x.copy()
        best_f = func(x)
        eval_count = 1

        for gen in range(100000):
            if eval_count >= max_evals:
                break
            if time.time() >= deadline:
                break

            # Adjust lam if needed
            if eval_count + lam > max_evals:
                lam = max(1, max_evals - eval_count)
                if lam < 1:
                    break
                mu = max(1, lam // 2)
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mueff = int(1 / np.sum(w**2))

            # Sample and evaluate offspring
            offspring = []
            for i in range(lam):
                if time.time() >= deadline:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                fy = func(y)
                eval_count += 1
                offspring.append((fy, y, z))

            if len(offspring) < mu:
                break

            offspring.sort(key=lambda t: t[0])

            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Update mean
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += w[i] * offspring[i][1]

            # Update ps and sigma
            y_mean = np.zeros(n)
            for i in range(mu):
                y_mean += w[i] * offspring[i][2]
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D * y_mean))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-15, 1e8)

            # Update pc and C
            y_inv = B @ (D ** -1) @ (x_new - x)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y_inv
            hsig = 1.0 if (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1))) < (1.4 + 1 / n) * chi_n) else 0.0
            outer = np.outer(y_inv, y_inv)
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * (outer + delta_hsig * np.outer(pc, pc))
            for i in range(mu):
                C += c_mu * w[i] ** 2 * np.outer(offspring[i][2], offspring[i][2])

            # Eigen decomposition
            try:
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.clip(eigval, 1e-20, None)
                D = np.sqrt(eigval)
                B = eigvec
            except:
                break

            x = x_new

        return best_x, best_f, eval_count

    def __call__(self, func):
        deadline = time.time() + 280
        rng = np.random.default_rng(42)
        d = self.dim

        best_x = np.zeros(d)
        best_f = np.inf
        total_evals = 0

        # Phase 1: Quick screening with diverse starts
        init_points = [
            np.zeros(d),
            np.ones(d) * 3.0,
            np.ones(d) * -3.0,
            rng.uniform(-4, 4, d),
            rng.uniform(-4, 4, d),
            rng.uniform(-4, 4, d),
        ]

        screen_budget = max(40, self.budget // 8)
        candidates = []

        for x0 in init_points:
            if time.time() >= deadline or total_evals >= self.budget:
                break
            x, f, used = self._cma_run(func, x0, 2.0, screen_budget, rng, deadline)
            total_evals += used
            candidates.append((f, x))

        candidates.sort(key=lambda t: t[0])

        if candidates:
            best_f, best_x = candidates[0]

        # Phase 2: Refine best candidate
        remaining = self.budget - total_evals
        if remaining > 20 and time.time() < deadline:
            x, f, used = self._cma_run(func, best_x, 0.3, remaining, rng, deadline)
            total_evals += used
            if f < best_f:
                best_f = f
                best_x = x

        if best_f == np.inf:
            best_x = np.zeros(d)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
