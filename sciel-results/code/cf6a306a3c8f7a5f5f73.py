"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Single-trajectory Rank-One CMA-ES with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        call_count = 0

        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        # Initial point
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-4, 4, dim)
        f0 = counted_func(x0)
        best_x = x0.copy()
        best_f = f0

        # CMA-ES parameters
        lambda_ = 4 * (dim + 2)  # population size
        c_sigma = (4 + dim / lambda_) / ((dim + 2) ** 2 + 4)
        c_c = (4 + dim / lambda_) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        # Precompute Cholesky of C for sampling
        L = np.linalg.cholesky(C)

        # Track progress for restart logic
        stagnation = 0
        best_f_history = [best_f]

        for iteration in range(budget // lambda_ + 10):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            for i in range(lambda_):
                if call_count >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = best_x + sigma * L @ z
                ys.append(y)

            if not ys:
                break

            # Evaluate population
            fs = []
            for y in ys:
                if call_count >= budget or time.time() > deadline:
                    break
                fy = counted_func(y)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(fs) < lambda_:
                break

            # Sort by fitness
            order = np.argsort(fs)
            y_sorted = [ys[i] for i in order]
            f_sorted = [fs[i] for i in order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Weighted mean
            m = np.zeros(dim)
            for i in range(mu):
                m += weights[i] * y_sorted[i]

            # Update evolution paths
            y_mean_diff = m - best_x
            best_x = m  # Update center of search

            # Sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_mean_diff
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e3)

            # Covariance matrix update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean_diff
            delta_heav = (1 + (1 - c_c) ** 2) * (1 - c_c) - (1 - c_c) ** 2 * mu_eff
            C = (1 + c_1 * delta_heav - c_1) * C + c_1 * np.outer(y_mean_diff, y_mean_diff) - c_1 / chi_n ** 2 * np.outer(p_c, p_c)

            # Ensure symmetry and positive semi-definite
            C = 0.5 * (C + C.T)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

            # Stagnation check
            if best_f < best_f_history[-1] - 1e-6:
                stagnation = 0
            else:
                stagnation += 1
            best_f_history.append(best_f)

            if stagnation > 30:
                break

        # Nelder-Mead polish with remaining budget
        remaining = budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            try:
                nm_result = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining, 200), 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = nm_result.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
