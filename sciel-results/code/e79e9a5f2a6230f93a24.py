"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lightweight CMA-ES for rotated Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin before 330s

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Initial step size
        sigma = 2.0
        # Mean (start near origin, but Rosenbrock minimum is at (1,...,1) before rotation)
        # Since rotation is hidden, start at origin
        m = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Step size control
        d_sigma = (np.sqrt(lam + 1) - 1) / (dim + 2) + 2 / (dim + lam - 2)
        # Update rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))
        # Eigen-decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)
        # Cholesky or sqrt of C
        C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T

        best_x = m.copy()
        best_f = float('inf')
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Initial evaluation
        f0 = eval_point(m)
        if f0 is None:
            return best_x, best_f

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            population = []
            fitnesses = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = eval_point(y)
                if fy is None:
                    break
                population.append(y)
                fitnesses.append(fy)

            if len(population) < 2:
                break

            # Sort by fitness
            idx = np.argsort(fitnesses)
            population = [population[i] for i in idx]
            fitnesses = [fitnesses[i] for i in idx]

            # Update mean
            new_m = np.sum(np.array(weights)[:, None] * np.array(population[:mu]), axis=0)

            # Update evolution paths
            # Inverse sqrt of C for ps
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_m - m) / sigma
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) / np.sqrt(mu_eff)
            if h_sigma < 1.4 + 2 / (dim + 1):
                pc = (1 - c_c) * pc + c_c * (new_m - m) / sigma
                C = (1 + c_c * (2 - c_c) * mu_eff - c_1) * C + c_1 * (pc @ pc - C)
                C += c_mu * np.sum(weights[:mu, None] * np.array(population[:mu] - m)[:, None] @ (population[:mu] - m), axis=0) / sigma ** 2
            else:
                C = (1 - c_1) * C + c_1 * (C + pc @ pc)

            # Update sigma
            sigma *= np.exp(d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update eigen-decomposition periodically
            if (calls + 1) % max(1, lam // 2) == 0:
                C = 0.5 * (C + C.T)  # symmetrize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            m = new_m

        return best_x, best_f
# EVOLVE-BLOCK-END
