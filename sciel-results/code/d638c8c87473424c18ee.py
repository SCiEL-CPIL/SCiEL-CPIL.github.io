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
        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        mu = lam // 2  # number of recombination weights
        # Weights (logarithmic)
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        x = np.zeros(dim)  # Start from origin
        sigma = 2.0
        B = np.eye(dim)
        D = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial population
        best_x = x.copy()
        best_fx = func(x)

        for gen in range(budget // lam):
            if time.time() > deadline:
                break

            # Generate population
            pop = []
            fvals = []
            for i in range(lam):
                # Sample from N(x, sigma^2 * B @ diag(D^2) @ B.T)
                z = rng.standard_normal(dim)
                y = x + sigma * (B * D) @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                pop.append(y)
                fvals.append(fy)

            # Sort by fitness
            idx = np.argsort(fvals)
            pop = np.array([pop[i] for i in idx])
            fvals = np.array([fvals[i] for i in idx])

            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pop[0].copy()

            # Recombination
            m = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update step size control (cs)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / (sigma * D) * (B @ (m - x))
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            damps = 1 + 2 * max(0, (mu_eff - 2) / (dim + 2) - 1) + cs

            # Update evolution path (cc)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) / (sigma * D) * (B @ (m - x))

            # Update covariance matrix
            delta_hs = (1 - hsig) * cc * (2 - cc)
            outer = np.outer(pc, pc)
            C_update = (1 + c1 * delta_hs - c1 - cmu) * np.eye(dim)
            C_update += c1 * outer
            for i in range(mu):
                zi = (pop[i] - x) / sigma
                C_update += cmu * weights[i] * np.outer(zi, zi)
            # B and D update (eigenvalue decomposition)
            eigvals, eigvecs = np.linalg.eigh(C_update)
            eigvals = np.maximum(eigvals, 1e-20)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update mean
            x = m

        return best_x, best_fx
# EVOLVE-BLOCK-END
