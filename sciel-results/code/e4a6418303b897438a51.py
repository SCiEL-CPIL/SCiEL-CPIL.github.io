"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive population and restarts for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(12345)
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma0, budget):
            nonlocal calls_made, best_x, best_f
            n = self.dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            # CMA-ES parameters
            c_sigma = (n + 2) / (n + 4)
            d_sigma = n / 2 + 1
            c_c = (4 + n / n) / (4 + n + n)
            c_mu = min(0.5, (2 + 2 / (n + 1.7)) / (n + 4 + 2 / (n + 1.7)))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

            # Initialize
            mean = x0.copy()
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)
            B = np.eye(n)
            d = np.ones(n)
            ps = np.zeros(n)
            phis = np.zeros(n)
            sigma = sigma0

            # Weights
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(0.5, (2 - c_1 + 2 * mu_eff / n) / ((n + 2) ** 2 + mu_eff))

            # Precompute
            for _ in range(budget):
                if calls_made >= self.budget or time_left() < 5:
                    break

                # Generate offspring
                xs = np.zeros((lam, n))
                fs = np.zeros(lam)
                for i in range(lam):
                    z = rng.standard_normal(n)
                    x = mean + sigma * (B @ (d * z))
                    x = np.clip(x, -5, 5)
                    xs[i] = x
                    fs[i] = evaluate(x)
                    if calls_made >= self.budget or time_left() < 5:
                        break

                # Sort
                idx = np.argsort(fs)
                xs_sorted = xs[idx]
                fs_sorted = fs[idx]

                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * xs_sorted[:mu], axis=0)
                new_mean = np.clip(new_mean, -5, 5)

                # Update step-size control
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (C_inv_sqrt @ (new_mean - mean)))
                h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made % 1000 + 1))) < (1 + 2 / (n + 1)) * chi_n else 0
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance matrix
                delta_mean = new_mean - mean
                rank_one = np.outer(delta_mean, delta_mean)
                rank_mu = np.sum(weights[:, np.newaxis] * np.outer(xs_sorted[:mu] - mean, xs_sorted[:mu] - mean), axis=0)
                C = (1 + c_1 * h_sigma * (n + 1) - c_1 - c_mu) * C + c_1 * (rank_one + (1 - h_sigma) * ps @ ps) + c_mu * rank_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update eigen-decomposition
                try:
                    d, B = np.linalg.eigh(C)
                    d = np.clip(d, 1e-20, None)
                    C_inv_sqrt = (B / np.sqrt(d)) @ B.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_inv_sqrt = np.eye(n)
                    B = np.eye(n)
                    d = np.ones(n)

                mean = new_mean

                # Track best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = xs_sorted[0].copy()

                # Stagnation check
                if np.std(fs_sorted[:mu]) < 1e-10 and np.linalg.norm(delta_mean) < 1e-10:
                    break

        # Phase 1: Initial exploration with multiple random starts
        n_starts = min(6, max(2, self.budget // 80))
        starts = [rng.uniform(-5, 5, self.dim) for _ in range(n_starts)]
        starts.append(np.zeros(self.dim))

        # Evaluate all starts first
        for s in starts:
            if calls_made >= self.budget or time_left() < 10:
                break
            evaluate(s)

        # Sort starts by fitness
        start_fitnesses = []
        for s in starts:
            # Re-evaluate to get current fitness (or use cached)
            pass

        # Phase 2: Run CMA-ES from best starts
        # Use top 3 starts
        top_starts = starts[:3]
        budget_per_start = max(30, (self.budget - calls_made - 20) // len(top_starts))

        for i, x0 in enumerate(top_starts):
            if calls_made >= self.budget or time_left() < 15:
                break
            remaining = min(budget_per_start, self.budget - calls_made - 10)
            if remaining < 10:
                break
            cma_es(x0, 1.5, remaining)

        # Phase 3: Final refinement from global best
        if best_x is not None and calls_made < self.budget and time_left() > 10:
            remaining = min(50, self.budget - calls_made)
            if remaining >= 5:
                cma_es(best_x.copy(), 0.3, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
