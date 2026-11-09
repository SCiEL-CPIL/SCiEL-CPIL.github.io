"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Hard wall-clock deadline with safety margin (evaluator kills at 330s).
        deadline = time.time() + 300.0

        # Strict call accounting: never exceed budget, never exceed deadline.
        calls = 0
        best_x = None
        best_f = np.inf

        def fcall(x):
            nonlocal calls, best_x, best_f
            # Deadline guard: if we're about to exceed the limit, stop.
            if time.time() > deadline:
                raise _StopSearch()
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        class _StopSearch(Exception):
            pass

        # --- CMA-ES core (operates on raw coordinates, clipped at evaluation) ---
        def cmaes(x0, sigma0, n_calls, pop):
            nonlocal calls
            n = dim
            # Standard CMA-ES parameters
            mu = max(1, pop // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / (w @ w)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)

            for it in range(n_calls):
                if time.time() > deadline:
                    raise _StopSearch()
                # Generate population
                Y = np.empty((pop, n))
                for i in range(pop):
                    z = rng.standard_normal(n)
                    Y[i] = x + sigma * (B @ (D * z))
                # Evaluate (with clipping + accounting)
                F = np.empty(pop)
                for i in range(pop):
                    F[i] = fcall(Y[i])
                # Sort
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]
                # Mean update
                y_w = w @ Y
                x_old = x
                x = x + sigma * y_w
                # Path updates
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma * chi_n) * (B @ (D * y_w))
                hsig = min(1, (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) / chi_n) < 1.4 + 2 / (n + 1))
                pc = (1 - c_c) * pc + hsig * c_c * y_w
                # Covariance update
                delta_h = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_1 - c_1 * n + delta_h) * C \
                    + c_1 * (np.outer(pc, pc) - C) \
                    + c_mu / sigma ** 2 * sum(w[i] * np.outer(Y[i] - x_old, Y[i] - x_old) for i in range(mu))
                # Eigen-decomposition (recompute adaptively; n=5 is cheap)
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-20))
                B = B2
                D = D2
                # Sigma update
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(sigma, 1e4)

        # Multi-start schedule: list of (budget, sigma) pairs.
        # First restart gets the most budget; later restarts are shorter and
        # start from random points to escape the secondary basin.
        schedule = [
            (300, 1.5),
            (100, 2.5),
            (50, 3.0),
            (30, 3.5),
            (20, 4.0),
        ]
        # Adjust to fit within budget
        total = sum(b for b, _ in schedule)
        if total > budget - 5:
            # Scale down proportionally, keep at least 10 each
            scale = (budget - 5) / total
            schedule = [(max(10, int(b * scale)), s) for b, s in schedule]

        for b, s in schedule:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x0 = rng.uniform(-5, 5, dim)
            try:
                cmaes(x0, s, b, pop=12)
            except _StopSearch:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = fcall(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
