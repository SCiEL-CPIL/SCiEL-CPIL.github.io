"""Rewritten optimizer: CMA-ES with adaptive step size and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive covariance and Nelder-Mead final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        # Initial sigma based on problem scale
        sigma = 1.0
        # Covariance matrix (identity start)
        C = np.eye(dim)
        # Evolution path for covariance
        p_sigma = np.zeros(dim)
        # Evolution path for mean shift
        p_c = np.zeros(dim)
        # Learning rates
        mu = max(1, dim // 2)  # number of parents
        w = np.zeros(mu)
        # Weights: log-linear
        w[0] = 1.0
        for i in range(1, mu):
            w[i] = w[i - 1] * 0.9
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Number of offspring
        lambda_pop = 4 * dim

        # Reserve last 30 evaluations for Nelder-Mead
        nm_budget = min(30, budget // 10)
        cma_budget = budget - nm_budget - 1  # -1 for initial evaluation

        # CMA-ES main loop
        for iteration in range(cma_budget):
            # Check time
            if time.time() > deadline:
                break

            # Generate lambda offspring
            ys = []
            f_ys = []
            for _ in range(lambda_pop):
                # Sample from multivariate normal with covariance C and step size sigma
                z = rng.multivariate_normal(np.zeros(dim), C)
                y = x + sigma * z
                # Clip to bounds
                y = np.clip(y, -5, 5)
                f_y = func(y)
                ys.append(y)
                f_ys.append(f_y)

            # Sort by fitness
            ys = np.array(ys)
            f_ys = np.array(f_ys)
            idx = np.argsort(f_ys)
            ys = ys[idx]
            f_ys = f_ys[idx]

            # Update best
            if f_ys[0] < best_fx:
                best_fx = f_ys[0]
                best_x = ys[0].copy()

            # Update mean using weighted combination of mu best
            w_sum = np.sum(w)
            x_new = np.sum(w[:, None] * ys[:mu], axis=0) / w_sum

            # Update evolution paths
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_sqrt * (x_new - x) / sigma
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1 - 1) / mu_eff else 0.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            h_c = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1 - 1) / mu_eff else 0.0

            # Update covariance
            delta_h = (1 - h_c) * c_c * (2 - c_c)
            C = (1 + c_1 * (delta_h - 1) - c_mu) * C
            C += c_1 * np.outer(p_c, p_c)
            C += c_mu * np.sum(w[:, None] * np.outer(ys[:mu] - x, ys[:mu] - x), axis=0) / w_sum

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization for numerical stability
            C += 1e-12 * np.eye(dim)

            x = x_new

            # Adapt sigma
            target_sigma = 1.0
            # Simple adaptation: adjust based on success rate
            success = np.sum(f_ys[:mu] < fx) / mu
            if success > 0.5:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = np.clip(sigma, 1e-8, 10.0)

        # Nelder-Mead polish phase
        if time.time() < deadline and nm_budget > 0:
            # Start from best point found
            x0 = best_x.copy()
            # Initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                point = x0.copy()
                step = 0.1 if abs(x0[i]) < 0.5 else 0.05
                point[i] += step
                point = np.clip(point, -5, 5)
                simplex.append(point)

            # Nelder-Mead with limited iterations
            def nm_func(v):
                if time.time() > deadline:
                    return best_fx
                return func(np.clip(v, -5, 5))

            # Use scipy minimize with Nelder-Mead
            try:
                result = minimize(nm_func, x0, method='Nelder-Mead',
                                  options={'maxiter': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10})
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
