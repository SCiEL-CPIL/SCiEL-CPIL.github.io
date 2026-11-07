"""CMA-ES based optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin: 330s kill, stop at 280s
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size: standard CMA-ES formula, at least 4
        lam = max(4, int(4 + 2 * np.log(dim)))
        # Ensure we don't exceed budget
        if lam * 10 > budget:
            lam = max(4, budget // 10)

        # Initial mean: origin is a reasonable prior (minimizer in [-4,4]^5)
        mean = np.zeros(dim)
        sigma = 2.0  # initial step size covering a good portion of [-5,5]

        # CMA-ES state
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Adaptation parameters
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + mu_eff - 3)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial evaluation
        x0 = np.clip(mean, -5, 5)
        f_best = func(x0)
        best_x = x0.copy()
        best_f = f_best
        calls = 1

        # Evaluate initial population
        def evaluate_population():
            nonlocal calls, best_x, best_f
            ys = []
            fvals = []
            for i in range(lam):
                zi = rng.standard_normal(dim)
                y = mean + sigma * (C_sqrt @ zi)
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fvals.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            return np.array(ys), np.array(fvals)

        # Initial population
        ys, fvals = evaluate_population()

        # Main loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            ys_sorted = ys[idx]
            fvals_sorted = fvals[idx]

            # Select best mu individuals
            y_selected = ys_sorted[:mu]
            w = weights

            # Update mean
            new_mean = np.sum(w[:, None] * y_selected, axis=0)
            # Keep mean in bounds
            new_mean = np.clip(new_mean, -5, 5)

            # Update step size
            z = (new_mean - mean) / (sigma + 1e-30)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt @ z)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lam))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update evolution path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / (sigma + 1e-30)

            # Update covariance
            delta_sigma = (1 + np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1) * c_sigma * (2 - c_sigma)
            C = ((1 - c_1 - c_mu + delta_sigma) * C
                 + c_1 * (np.outer(pc, pc) - C)
                 + c_mu * np.sum(w[:, None, None] * np.outer(y_selected - mean, y_selected - mean), axis=0))

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize
            C += 1e-10 * np.eye(dim)

            # Update C_sqrt via eigen-decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

            mean = new_mean

            # Generate next population
            ys, fvals = evaluate_population()

            # Check deadline again
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
