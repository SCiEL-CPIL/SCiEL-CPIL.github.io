"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * dim)  # population size
        mu = lambda_ // 2  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if calls >= total_budget:
                return None
            calls += 1
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    return None
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        # Initial population
        L = np.linalg.cholesky(C)
        pop_x = []
        pop_f = []
        for i in range(lambda_):
            z = np.random.randn(dim)
            x_i = x_mean + sigma * (L @ z)
            f_i = safe_call(x_i)
            if f_i is None:
                return best_x, best_f
            pop_x.append(x_i)
            pop_f.append(f_i)

        # Main loop
        while calls < total_budget:
            if time.time() > deadline:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Update mean
            x_new = sum(weights[i] * pop_x[i] for i in range(mu))

            # Update step size
            y_mean = (x_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance
            pc = (1 - c_c) * pc + c_c * (y_mean if np.linalg.norm(y_mean) < (1.4 + 1 / c_c) * chi_n else 0)
            delta_heaviside = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / lambda_)) < (1.4 + 1 / c_c) * chi_n else 0
            C = (1 + c_1 * delta_heaviside - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + delta_heaviside - 1) \
                + c_mu * sum(weights[i] * np.outer(pop_x[i] - x_mean, pop_x[i] - x_mean) for i in range(mu)) / sigma ** 2

            # Ensure symmetry and positive definiteness
            C = 0.5 * (C + C.T)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)

            x_mean = x_new
            sigma = max(sigma, 1e-12)

            # Generate new population
            pop_x = []
            pop_f = []
            for i in range(lambda_):
                if calls >= total_budget:
                    break
                if time.time() > deadline:
                    return best_x, best_f
                z = np.random.randn(dim)
                x_i = x_mean + sigma * (L @ z)
                f_i = safe_call(x_i)
                if f_i is None:
                    return best_x, best_f
                pop_x.append(x_i)
                pop_f.append(f_i)

            if len(pop_f) < mu:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
