"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES style optimizer with adaptive population."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 280  # Safety margin before 300s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            if x.shape != (dim,):
                x = x.reshape(dim)
            f = float(func(x))
            calls_made += 1
            return f

        # Strategy: multiple restarts with CMA-ES-like adaptation
        # For dim=5, use population size ~ 4*dim = 20
        # Budget 500, so we can do ~25 generations per restart
        # Use 2-3 restarts to escape local minima

        num_restarts = 3
        budget_per_restart = budget // num_restarts

        for restart in range(num_restarts):
            if time.time() > self.deadline or calls_made >= budget:
                break

            # Initialize center
            if restart == 0:
                center = rng.uniform(-4, 4, dim)
            else:
                # Start near previous best with some noise
                center = best_x + rng.normal(0, 1.0, dim)
                center = np.clip(center, -4.5, 4.5)

            # CMA-ES parameters
            mu = dim // 2  # number of parents
            lambda_ = 4 * dim  # population size
            # Weights
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            # Step size
            sigma = 1.5 / (np.sqrt(2 * dim) * (1 + 0.5 * np.log(dim)))
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_C = np.zeros(dim)
            # Damping
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            # Inverse square root of C (initially identity)
            # We'll use Cholesky or eigen decomposition periodically

            # Generation loop
            max_gen = budget_per_restart // lambda_
            for gen in range(max_gen):
                if time.time() > self.deadline or calls_made >= budget:
                    break

                # Generate population
                # Update inverse square root of C (simplified: use eigen decomposition)
                # For dim=5, this is cheap
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Sample candidates
                ys = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = center + sigma * (D_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    ys.append(y)

                # Evaluate
                fs = np.array([safe_call(y) for y in ys])
                if calls_made >= budget:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                ys_sorted = np.array(ys)[idx]
                fs_sorted = fs[idx]

                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = ys_sorted[0].copy()

                # Compute new center (weighted mean)
                new_center = np.sum(w[:, None] * ys_sorted[:mu], axis=0)

                # Update evolution paths
                # p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                    D_inv_sqrt.T @ (new_center - center) / sigma
                )
                # h_sigma
                chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim))
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / dim) * chi_n:
                    h_sigma = 1
                else:
                    h_sigma = 0

                # p_C
                p_C = (1 - c_c) * p_C + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                    D_inv_sqrt.T @ (new_center - center) / sigma
                )

                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank1 = np.outer(p_C, p_C)
                rankmu = np.sum(
                    w[:, None, None] * np.outer(ys_sorted[:mu] - center, ys_sorted[:mu] - center),
                    axis=0
                ) / (sigma ** 2)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu

                # Ensure positive definite
                eigvals_c, _ = np.linalg.eigh(C)
                if np.any(eigvals_c < 1e-14):
                    C = C + 1e-14 * np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update center
                center = new_center

        # Final: use remaining budget for local (1+1)-ES refinement
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - calls_made
            for _ in range(remaining):
                if time.time() > self.deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                if calls_made >= budget:
                    break
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
