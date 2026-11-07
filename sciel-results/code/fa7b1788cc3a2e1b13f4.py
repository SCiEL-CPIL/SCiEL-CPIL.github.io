import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf

        # Phase 1: Quick multi-start to find a good region
        # Use 10 initial points spread across the domain
        n_init = min(10, total_budget)
        init_points = []
        for i in range(n_init):
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            init_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness
        init_points.sort(key=lambda t: t[1])

        # Phase 2: CMA-ES from the best starting point
        # Use remaining budget
        remaining = total_budget - n_init

        # CMA-ES parameters
        lambda_pop = 10  # population size
        mu = lambda_pop // 2  # number of selected individuals

        # Start from best initial point
        x_mean = best_x.copy()
        f_mean = best_f

        # Initial sigma: use spread of initial points or default
        # Estimate scale from the initial points
        if n_init > 1:
            spreads = np.linalg.norm(np.array([p[0] for p in init_points[:5]]) - x_mean, axis=1)
            sigma = max(0.5, np.mean(spreads) * 0.5)
        else:
            sigma = 1.0

        # CMA-ES state
        C = np.eye(dim)
        # Step size adaptation
        c_sigma = (mu + 0.3) / (dim + 2)
        d_sigma = 1 + 2 * max(0, (sqrt(dim) - 1) / (dim + 2))  # approximation
        # Actually let's use standard CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 0.5) / (dim + 3)
        d_sigma = 1 + 2 * max(0.0, (sqrt(mu_eff) - 1) / (dim + 2)) if dim > 1 else 1.0
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        pc = np.zeros(dim)
        ps = np.zeros(dim)

        evals_used = n_init
        stagnation_counter = 0
        best_in_run = best_f

        # Main CMA-ES loop
        while evals_used < total_budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((lambda_pop, dim))
            # Whitening transformation
            # Use Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            Y = np.array([B[i] @ L for i in range(lambda_pop)])
            X = x_mean + sigma * Y
            # Clip to bounds
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(X[i]) for i in range(lambda_pop)])
            evals_used += lambda_pop

            # Check for new best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]

            # Update mean
            x_new_mean = np.sum(weights[:, None] * X[:mu], axis=0)

            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ x_new_mean - x_mean)
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n < (1 + 1.4 / dim) * 1.0 else 0

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update covariance
            pc = (1 - c_c) * pc + c_c * (x_new_mean - x_mean) / sigma
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(X[:mu] - x_mean, X[:mu] - x_mean) / sigma ** 2, axis=0)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(dim)

            x_mean = x_new_mean

            # Stagnation detection: if no improvement for many generations, restart
            if stagnation_counter > 15:
                # Restart from best point with reduced sigma
                x_mean = best_x.copy()
                sigma = max(0.1, sigma * 0.3)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                stagnation_counter = 0

            # If sigma is too small, restart
            if sigma < 1e-6:
                x_mean = best_x.copy()
                sigma = 0.5
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                stagnation_counter = 0

        return best_x, best_f
