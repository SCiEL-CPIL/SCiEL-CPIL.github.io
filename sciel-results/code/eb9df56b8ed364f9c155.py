"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            return f

        # Initial exploration to find a good starting point
        init_budget = max(10, budget // 10)
        for i in range(init_budget):
            if calls_made >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if calls_made >= budget:
            return best_x, best_f

        # CMA-ES parameters
        lambda_ = max(4, 4 + int(3 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        x = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        c_sigma = (lambda_ + 2) / (dim + 4)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        remaining = budget - calls_made
        max_iter = remaining // lambda_
        
        # Cholesky decomposition for sampling
        try:
            B = np.linalg.cholesky(C)
        except:
            B = np.eye(dim)

        for it in range(max_iter):
            if calls_made + lambda_ > budget:
                lam_this = budget - calls_made
                if lam_this <= 0:
                    break
            else:
                lam_this = lambda_

            # Generate population
            ys = []
            for i in range(lam_this):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ z)
                y = np.clip(y, -5.0, 5.0)
                ys.append(y)

            # Evaluate
            fvals = np.array([safe_call(y) for y in ys])

            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = [ys[i] for i in order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = ys_sorted[0].copy()

            # Compute weighted mean
            w = weights[:min(mu, lam_this)]
            if len(w) < min(mu, lam_this):
                w = np.ones(min(mu, lam_this)) / min(mu, lam_this)
            m = sum(w[i] * ys_sorted[i] for i in range(len(w)))

            # Update evolution paths
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_sqrt * (m - x) / (sigma + 1e-30)

            # Update covariance matrix
            c_c_sqrt = np.sqrt(c_c * (2 - c_c) * mu_eff)
            p_c = (1 - c_c) * p_c + c_c_sqrt * (m - x)

            # Update mean
            x = m

            # Update covariance
            delta_h = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) - 1) * d_sigma * c_sigma
            C = (1 + c_c * (2 - c_c) * delta_h) * C
            for i in range(len(w)):
                z = (ys_sorted[i] - x) / (sigma + 1e-30)
                C += 2 * c_c * w[i] * np.outer(z, z)

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(0.01, min(sigma, 10.0))

            # Update Cholesky
            try:
                B = np.linalg.cholesky(C)
            except:
                B = np.eye(dim)

            # Early stopping if progress is negligible
            if it > 5 and np.all(np.abs(np.array(ys_sorted[:3]) - x) < 1e-6):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
