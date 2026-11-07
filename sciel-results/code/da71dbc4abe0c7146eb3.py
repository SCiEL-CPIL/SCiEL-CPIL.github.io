"""Optimizer for BBOB f12 (Bent Cigar) using CMA-ES with adaptive restarts."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        d = self.dim
        total_budget = self.budget
        calls = [0]

        def eval_point(x):
            x = np.clip(x, -5, 5)
            f = func(x)
            calls[0] += 1
            return f

        best_x = np.array([0.0] * d)
        best_fx = np.inf

        # CMA-ES parameters
        def run_cmaes(x0, sigma0, budget_remaining):
            nonlocal best_x, best_fx, calls
            if budget_remaining <= 0:
                return best_x, best_fx

            # Population size
            lam = 4 + int(3 * np.log(d))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            cs = (mueff + 2) / (d + mueff + 5)
            c1 = (2 - mueff / (d + 1) + 2 / (d + 1)) / ((d + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))

            # State
            xmean = x0.astype(float).copy()
            sigma = sigma0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            chiN = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (20 * d ** 2))

            # Evaluate initial point
            fx0 = eval_point(xmean)
            if fx0 < best_fx:
                best_fx = fx0
                best_x = xmean.copy()

            budget_used = 1

            while budget_used < budget_remaining and calls[0] < total_budget:
                if time.time() > deadline:
                    break

                # Generate population
                B = np.linalg.cholesky(C)
                z = np.random.randn(lam, d)
                y = z @ B.T
                x_pop = xmean + sigma * y
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate
                f_pop = np.array([eval_point(x) for x in x_pop])
                budget_used += lam
                if calls[0] >= total_budget:
                    break

                # Sort
                order = np.argsort(f_pop)
                x_pop = x_pop[order]
                f_pop = f_pop[order]

                # Update best
                if f_pop[0] < best_fx:
                    best_fx = f_pop[0]
                    best_x = x_pop[0].copy()

                # Compute new mean
                xmean_new = sum(weights[i] * x_pop[i] for i in range(mu))

                # Update CMA state
                y_mean = (xmean_new - xmean) / sigma if sigma > 1e-15 else np.zeros(d)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (budget_used + 1))) < (
                        (1 + 2 / (d + 1)) * chiN
                )
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_mean

                # Update covariance
                delta_d = (1 + hsig * (cc / (1 - cc)) - 1) if hsig else 1.0
                outer = np.outer(y_mean, y_mean)
                C = (1 + c1 * delta_d - c1 - cmu) * C + cmu * sum(
                    weights[i] * np.outer(x_pop[i] - xmean, x_pop[i] - xmean) / (sigma ** 2)
                    for i in range(mu)
                )

                # Update sigma
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))

                xmean = xmean_new
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 50.0)

            return best_x, best_fx

        # Multi-start CMA-ES
        rng = np.random.default_rng(12345)
        n_starts = 5
        budget_per_start = total_budget // n_starts

        for s in range(n_starts):
            if calls[0] >= total_budget or time.time() > deadline:
                break

            if s == 0:
                x0 = np.zeros(d)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-3, 3, d)
                sigma0 = 1.5

            remaining = total_budget - calls[0]
            if remaining <= 0:
                break

            run_cmaes(x0, sigma0, remaining)

        return best_x, best_fx
