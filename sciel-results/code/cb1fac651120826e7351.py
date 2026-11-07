"""CMA-ES variant for BBOB f20 (Schwefel) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with small population for rotated Schwefel landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        n = self.dim
        lam = 10  # population size
        mu = lam // 2  # number of parents for update

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # CMA-ES parameters
        cmu = 2 * (2 / (n + 4))**2 - 1
        c1 = (4 + ln / (n + 1)) / ((n + 2)**2 + 4)
        cmu = 2 * (min(mu, n) / (n + 2))**2 - 1
        cmu = min(cmu, 1 - c1 - 2 * (1 - c1) / (n + 2))
        c_sigma = (mu + 2) / (n + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # Success history
        hsig = 0.0
        hsig = 0.0

        # Evaluate initial point
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Generate initial population
        def generate_population():
            B = np.linalg.cholesky(C)
            ys = rng.standard_normal((lam, n))
            return x + sigma * ys @ B.T

        def sort_by_fitness(ind, vals):
            order = np.argsort(vals)
            return [ind[i] for i in order], [vals[i] for i in order]

        for gen in range(self.budget // lam):
            if time.time() > deadline:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            ys = rng.standard_normal((lam, n))
            pops = x + sigma * (ys @ B.T)

            # Clip to box
            pops = np.clip(pops, -5, 5)

            # Evaluate
            vals = np.array([func(p) for p in pops])

            # Update best
            min_idx = np.argmin(vals)
            if vals[min_idx] < best_f:
                best_f = vals[min_idx]
                best_x = pops[min_idx].copy()

            # Sort
            order = np.argsort(vals)
            sorted_pops = pops[order]
            sorted_vals = vals[order]

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1 / np.sum(w**2)

            # Update mean
            x_new = sum(w[i] * sorted_pops[i] for i in range(mu))

            # Update evolution paths
            pc = (1 - c1) * pc + c1 * (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (x_new - x) / sigma

            # Update covariance
            delta_sigma = (np.linalg.norm(ps) / chi_n - 1) * d_sigma
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + \
                cmu / sigma**2 * sum(
                    w[i] * np.outer(sorted_pops[i] - x, sorted_pops[i] - x)
                    for i in range(mu)
                )

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)

            # Symmetrize C
            C = (C + C.T) / 2

            x = x_new

        # Final polish with (1+1)-ES if time remains
        if time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while time.time() < deadline - 1:
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    best_x = x.copy()
                    best_f = fx
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                if sigma < 1e-8:
                    break

        return best_x, best_f
