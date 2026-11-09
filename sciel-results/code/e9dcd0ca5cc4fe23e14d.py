"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f6 with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        rng = np.random.default_rng(42)
        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Quick multi-start to find a good initial center
        n_starts = 8
        starts = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        for s in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            x = np.clip(s, -5, 5)
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Full CMA-ES
        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # lambda
        pop_size = max(pop_size, 6)
        mu = pop_size // 2  # number of parents
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        mean = best_x.copy()
        sigma = 1.0  # initial step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        while calls[0] < budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []

            for i in range(pop_size):
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Sample from N(0, C) using Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                z = rng.standard_normal(dim)
                y = L @ z
                x = np.clip(mean + sigma * y, -5, 5)
                try:
                    fx = safe_func(x)
                except StopIteration:
                    break
                ys.append(y)
                xs.append(x)
                fs.append(fx)

            if not xs:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            # Weighted mean of best mu solutions
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * xs[i]

            # Update evolution path for covariance
            pc = (1 - c_c) * pc + c_c * (new_mean - mean) / (sigma * chi_n)

            # Update evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff / C) * (new_mean - mean) / sigma

            # Stop if sigma is too small
            if sigma < 1e-12:
                break

            # Update covariance matrix
            delta = (1 + c_1 - c_mu) if np.dot(ps, ps) <= (1.4 + 1 / dim) * (1 - c_sigma) ** 2 else (1 + c_1 - c_mu + c_mu)
            C = (1 + delta * c_1 - c_mu) * C + c_mu * np.outer(ps, ps) - c_1 * (np.outer(pc, pc) - np.eye(dim))

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-10:
                    C += (1e-10 - eigvals.min()) * np.eye(dim)
            except:
                C = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            mean = new_mean

            # Adapt sigma based on progress
            if fs[0] < best_f * 0.999 or (best_f > 0 and fs[0] < best_f - 0.01):
                pass  # Good progress, keep sigma

        return best_x, best_f
# EVOLVE-BLOCK-END
