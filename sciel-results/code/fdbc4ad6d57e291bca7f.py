"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and time guard for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        lambda_ = 4 * dim  # population size

        # CMA-ES weights
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        # Precompute C^-1/2 and C^1/2 initially (identity)
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evaluate initial population
        def evaluate_population(x_center, sigma_cur, C_inv_sqrt_cur):
            """Generate and evaluate population."""
            ys = np.empty((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x_center + sigma_cur * (C_inv_sqrt_cur @ z)
                ys[i] = np.clip(y, -5, 5)
            fvals = np.array([func(ys[i]) for i in range(lambda_)])
            return ys, fvals

        # Initial population
        ys, fvals = evaluate_population(x, sigma, C_inv_sqrt)
        order = np.argsort(fvals)
        ys_sorted = ys[order]
        fvals_sorted = fvals[order]

        # Update best
        if fvals_sorted[0] < best_fx:
            best_fx = fvals_sorted[0]
            best_x = ys_sorted[0].copy()

        # Mean update
        x_new = (weights @ ys_sorted.T).T

        # Success rule for sigma
        # Simple adaptive step size based on improvement
        # Use CMA-ES update rules

        for iteration in range(1, budget // lambda_ + 1):
            if time.time() > deadline:
                break

            # Compute mean
            x_mean = (weights @ ys_sorted.T).T

            # Update evolution paths
            y_sigma = (x_mean - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ y_sigma)

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (ys_sorted[i] - x) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive semi-definite
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Update C^{-1/2} and C^{1/2}
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                C_sqrt = np.eye(dim)

            # Update center
            x = x_mean

            # Evaluate new population
            ys, fvals = evaluate_population(x, sigma, C_inv_sqrt)
            order = np.argsort(fvals)
            ys_sorted = ys[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = ys_sorted[0].copy()

        # Final polish with coordinate descent
        if time.time() < deadline - 10:
            x_polish = best_x.copy()
            fx_polish = best_fx
            for coord in range(100):
                improved = False
                for d in range(dim):
                    for delta in [0.1, 0.01, 0.001]:
                        for sign in [-1, 1]:
                            x_try = x_polish.copy()
                            x_try[d] = np.clip(x_polish[d] + sign * delta, -5, 5)
                            fx_try = func(x_try)
                            if fx_try < fx_polish:
                                x_polish = x_try
                                fx_polish = fx_try
                                improved = True
                if not improved:
                    break

            if fx_polish < best_fx:
                best_fx = fx_polish
                best_x = x_polish

        return best_x, best_fx
# EVOLVE-BLOCK-END
