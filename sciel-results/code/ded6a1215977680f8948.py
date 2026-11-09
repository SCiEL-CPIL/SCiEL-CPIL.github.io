"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        x = np.zeros(n)  # Start at origin (shifted problem, but symmetric-ish)
        sigma = 1.0
        # Covariance matrix C = I
        C = np.eye(n)
        # Eigen decomposition for sampling
        D = np.eye(n)
        B = np.eye(n)
        # Step size control
        cs = 1 / (n + 4)
        ds = 1 + 2 * max(0, (n - 1) / (n + 2) - 1) + cs
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 - (n + 1) / (n + 3)) / ((n + 2) ** 2 + 5))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        ps = np.zeros(n)
        pc = np.zeros(n)
        best_x = x.copy()
        best_f = func(x)

        # Population size
        lam = 4 + int(3 * np.log(n))
        w = np.array([np.log(np.log(lam + 1) - np.log(i + 1) + 1) for i in range(lam)])
        w = w / w.sum()
        mueff = int(np.floor(0.5 * lam))
        mu = w[:mueff].sum()
        w = w / w.sum()
        mueff = 1 / np.sum(w[:mueff] ** 2)

        calls_made = 1
        while calls_made < budget:
            # Generate population
            pop = []
            fvals = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D @ z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                pop.append(y)
                fvals.append(fy)
                calls_made += 1
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            pop = [pop[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # Update mean
            x_new = sum(w[i] * pop[i] for i in range(mueff))

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ (D @ ((x_new - x) / sigma)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (budget - calls_made)))) / chiN < (1.4 + 2 / (n + 1))
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (x_new - x) / sigma

            # Update covariance
            deltaD = (hsig - 1) * cs * ds
            C = (1 + c1 * deltaD - c1 - cmu) * C + c1 * np.outer(pc, pc)
            for i in range(mueff):
                y_i = (pop[i] - x) / sigma
                C += cmu * w[i] * np.outer(B @ (D @ y_i), B @ (D @ y_i))

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))

            # Update mean
            x = x_new

            # Update eigendecomposition
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(np.diag(D), 1e-12))
            # Reorder
            idx = np.argsort(np.diag(D))[::-1]
            B = B[:, idx]
            D = D[idx]

        return best_x, best_f
# EVOLVE-BLOCK-END
