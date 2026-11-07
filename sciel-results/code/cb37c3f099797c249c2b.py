"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # population size ~12 for dim=5
        mu = lam // 2  # number of recombination weights
        # Logarithmic weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize
        x = np.zeros(dim)
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)

        # Population initialization
        pop = [x + sigma * rng.standard_normal(dim) for _ in range(lam)]
        pop = [np.clip(p, -5, 5) for p in pop]
        fvals = [func(p) for p in pop]
        best_idx = int(np.argmin(fvals))
        x_best = pop[best_idx].copy()
        fx_best = fvals[best_idx]

        calls_made = lam
        budget_remaining = budget - calls_made
        gen = 0

        while budget_remaining > 0:
            if time.time() > deadline:
                break
            gen += 1

            # Sort by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < fx_best:
                fx_best = fvals[0]
                x_best = pop[0].copy()

            # Mean recombination
            m = sum(w[i] * pop[i] for i in range(mu))

            # Update evolution paths
            y = (m - x) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D * y))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * y))

            x = m

            # Update covariance matrix
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen)) / chi_n - 1
            delta_hs = (1 - hsig ** 2) * (1 + c_c)
            
            # Update C
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) - np.eye(dim))
            for i in range(mu):
                C += (c_mu * w[i] / mu_eff) * np.outer(pop[i] - x, pop[i] - x)

            # Periodic eigen-decomposition
            if gen % 10 == 0:
                D2, V = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-12))
                B = V
                D = D2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)

            # Generate new population
            new_pop = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                p = x + sigma * (B @ (D * z))
                p = np.clip(p, -5, 5)
                new_pop.append(p)

            new_fvals = [func(p) for p in new_pop]
            calls_made += lam
            budget_remaining = budget - calls_made

            pop = new_pop
            fvals = new_fvals

        return x_best, fx_best
# EVOLVE-BLOCK-END
