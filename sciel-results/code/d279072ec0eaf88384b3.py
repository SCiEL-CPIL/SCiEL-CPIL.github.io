"""Rewritten optimizer: multi-start CMA-ES with adaptive restarts for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Track best solution globally
        best_x = np.clip(rng.uniform(-5, 5, self.dim), -5, 5)
        best_f = func(best_x)
        eval_count = 1

        def cma_es(x0, sigma0, max_evals, rng, deadline):
            """Run CMA-ES from x0 with initial sigma0, up to max_evals or deadline."""
            nonlocal best_x, best_f, eval_count

            n = self.dim
            lam = 4 * int(1 + 3 * np.log(n))
            
            x = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Weights
            w = np.log(lam / 2) - np.log(np.arange(1, lam + 1))
            w = w / w.sum()
            mu = int(np.sum(w > 0))
            w = w[:mu]
            w = w / w.sum()
            mueff = 1 / np.sum(w ** 2)

            # Adaptation rates
            cc = (4 + lam / n) / (n + 4 + 2 * lam / n)
            cs = (lam / 2 + 1) / (n + lam / 2 + 3)
            cmu = min(2 - cc + 2 / (n - 1), 0.5 + 2 / (n ** 2 - 1))
            damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            iters = 0
            while eval_count < max_evals and time.time() < deadline:
                # Generate population
                z = rng.standard_normal((lam, n))
                B = np.linalg.cholesky(C)
                pop = np.array([x + sigma * (z[i] @ B.T) for i in range(lam)])
                pop = np.clip(pop, -5, 5)

                # Evaluate
                fvals = np.array([func(pop[i]) for i in range(lam)])
                eval_count += lam

                # Update best
                idx = np.argmin(fvals)
                if fvals[idx] < best_f:
                    best_f = fvals[idx]
                    best_x = pop[idx].copy()

                # Sort
                order = np.argsort(fvals)
                y = [pop[order[i]] for i in range(mu)]

                # Mean update
                x_old = x.copy()
                x = x + sigma * np.sum([w[i] * (y[i] - x_old) for i in range(mu)], axis=0)

                # Path updates
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (x - x_old) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iters + 1)))) / chi_n < 1.4 + 2 / (n + 1)
                pc = (1 - cc) * pc + hsig * cc * (x - x_old) / sigma

                # Covariance update
                C = (1 + cs * damps - cs - cmu) * C
                for i in range(mu):
                    C += cmu * w[i] * np.outer(y[i] - x_old, y[i] - x_old)
                C = (C + C.T) / 2

                # Ensure positive definiteness
                try:
                    np.linalg.cholesky(C)
                except:
                    C = C + 1e-5 * np.eye(n)

                # Sigma update
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))

                iters += 1

            return x, sigma, fvals[order[0]]

        # Multi-start strategy
        num_starts = 4
        evals_per_start = self.budget // num_starts

        for s in range(num_starts):
            if eval_count >= self.budget or time.time() >= deadline:
                break

            if s == 0:
                x0 = best_x.copy()
                sigma0 = 2.0
            elif s == 1:
                x0 = np.zeros(self.dim)
                sigma0 = 1.5
            elif s == 2:
                x0 = rng.uniform(-3, 3, self.dim)
                sigma0 = 1.0
            else:
                x0 = best_x + 0.5 * rng.standard_normal(self.dim)
                sigma0 = 0.3

            x0 = np.clip(x0, -5, 5)
            remaining = self.budget - eval_count
            max_evals_this = min(evals_per_start, remaining)

            if max_evals_this < 10:
                break

            cma_es(x0, sigma0, eval_count + max_evals_this, rng, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
