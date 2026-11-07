"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with random restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin below 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if len(x) != dim:
                x = np.resize(x, dim)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Random restarts to find a good basin
        n_restarts = min(20, budget // 10)
        for i in range(n_restarts):
            x = rng.uniform(-5, 5, dim)
            f = safe_call(x)
            if calls >= budget or time.time() > deadline:
                break

        if best_x is None:
            x = np.zeros(dim)
            safe_call(x)
            return best_x, best_f

        # Phase 2: CMA-ES from the best point
        x0 = best_x.copy()
        sigma = 1.0  # initial step size
        calls_before = calls

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        c_mu = (2 * mu_eff + 1) / ((dim + 2) ** 2 + mu_eff)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (1 - c_1) * min(mu_eff, 2) / (dim + 2)) / ((dim + 2) ** 2 + mu_eff)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        xmean = x0.copy()

        def cmaes_step():
            nonlocal xmean, C, pc, ps, sigma, calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return False
            # Generate population
            L = np.linalg.cholesky(C)
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = xmean + sigma * (L @ z)
                offspring.append((y, i))
            # Evaluate
            offs_f = []
            for y, i in offspring:
                f = safe_call(y)
                offs_f.append(f)
                if calls >= budget or time.time() > deadline:
                    break
            if len(offs_f) < lam:
                return False
            # Sort
            order = np.argsort(offs_f)
            # Update mean
            xnew = np.sum([np.array(offspring[order[i]][0]) * weights[i] for i in range(mu)], axis=0)
            # Update step size
            ymean = (xnew - xmean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (L.T @ ymean)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 1 / dim) * chi_n:
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            # Update C
            pc = (1 - c_c) * pc + c_c * (xnew - xmean) / sigma
            delta = c_1 * (np.outer(pc, pc) + (1 if np.linalg.norm(ps) ** 2 / chi_n ** 2 > (1.1 + 2 / (dim + 1)) * (1 - c_c) else 0) * np.eye(dim))
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = (np.array(offspring[order[i]][0]) - xmean) / sigma
                rank_mu += weights[i] * np.outer(yi, yi)
            C = (1 + c_1 * (1 - c_1) - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 if np.linalg.norm(ps) ** 2 / chi_n ** 2 > (1.1 + 2 / (dim + 1)) * (1 - c_c) else 0) * np.eye(dim)) + c_mu * rank_mu
            # Symmetrize
            C = (C + C.T) / 2
            xmean = xnew
            return True

        # Run CMA-ES
        while calls < budget and time.time() < deadline:
            if not cmaes_step():
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
