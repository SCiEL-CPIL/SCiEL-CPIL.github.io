"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time-aware budget and sigma adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        sigma = 1.0
        # Covariance matrix C (identity initially)
        C = np.eye(dim)
        # Step sizes per coordinate (for diagonal adaptation)
        step = np.ones(dim)

        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        mu = lambda_ // 2  # number of parents
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Eigendecomposition for C (start with identity)
        D = np.ones(dim)
        B = np.eye(dim)

        while calls < self.budget and time.time() < deadline:
            # Determine how many evaluations we can do in this generation
            # to avoid overshooting the budget
            remaining = self.budget - calls
            lambda_current = min(lambda_, remaining)
            
            if lambda_current < 2:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lambda_current):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls += 1

                # Update best
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

                if time.time() > deadline:
                    break

            # Sort by fitness
            if len(ys) < mu:
                break

            idx = np.argsort(fs)[:mu]
            ys_sorted = [ys[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]

            # Update mean
            x_old = x.copy()
            x = sum(w[i] * ys_sorted[i] for i in range(mu))

            # Update evolution path ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * ((x - x_old) / sigma)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((calls + 1) // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution path pc
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu / sigma**2 * sum(w[i] * np.outer(ys_sorted[i] - x_old, ys_sorted[i] - x_old) for i in range(mu))

            # Eigendecomposition
            try:
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-12))
                D = D2
                B = B2
            except np.linalg.LinAlgError:
                # Reset to identity if eigen decomposition fails
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)

        return best_x, best_fx
# EVOLVE-BLOCK-END
