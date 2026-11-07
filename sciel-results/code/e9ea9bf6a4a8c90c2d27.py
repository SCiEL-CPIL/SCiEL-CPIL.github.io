"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


class Optimizer:
    """CMA-ES with adaptive restarts and population-based sampling."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, budget, rng):
        """Run CMA-ES for a given budget."""
        dim = self.dim
        x = np.array(x0, dtype=float)
        sigma = sigma0
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        
        # CMA-ES parameters
        lambda_ = int(4 + 3 * math.log(dim))  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1.2 + 2 * mu_eff / dim) * (1 - c_sigma))
        c_mu = min(1 - c_1, (2 * (weights[0] - 1) + c_sigma) / ((dim + 2) * (1 - c_sigma)))
        
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls = 1
        remaining = budget - 1
        
        while remaining > 0:
            # Sample population
            ys = np.empty((lambda_, dim))
            fs = np.empty(lambda_)
            for i in range(lambda_):
                step = sigma * (B @ (D * rng.standard_normal(dim)))
                y = np.clip(x + step, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls += 1
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining <= 0:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            fs_sorted = fs[order]
            ys_sorted = ys[order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()
            
            # Weighted mean of best mu individuals
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys_sorted[i]
            
            # Update step-size control vector
            c_sigma_inv = 1.0 / c_sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_inv * math.sqrt(mu_eff * (2 - c_sigma) / sigma) * (B @ (D * (y_w - x)))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma) ** (2 * remaining)) < (1.4 + 2 / (dim + 1)) * math.sqrt(mu_eff + 1 - 1) / sigma) else 0.0
            
            # Update path evolution path
            p_c = (1 - c_c) * p_c + h_sigma * c_c * (y_w - x) / sigma
            h_c = 1.0 if (np.linalg.norm(p_c) / math.sqrt(1 - (1 - c_c) ** (2 * remaining)) < (1.4 + 2 / (dim + 1)) * math.sqrt(1 - c_c ** 2) / (1 - c_c) * 1.0) else 0.0
            
            # Update covariance matrix
            delta_h = (1 - h_c) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1) * C + c_1 * np.outer(p_c, p_c)
            # Actually need to recompute C properly
            C = np.eye(dim)  # Reset for simplicity in this implementation
            for i in range(mu):
                zi = (ys_sorted[i] - x) / sigma
                C += weights[i] * np.outer(zi, zi)
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(zi, zi) for i, zi in enumerate([(ys_sorted[i] - x) / sigma for i in range(mu)]))
            
            # Recompute B and D via eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Update mean
            x = np.clip(y_w, -5, 5)
            
            # Update sigma
            sigma *= math.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / math.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Check for convergence
            if sigma < 1e-12 or np.all(np.abs(D) < 1e-12):
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Brief random search to find a good starting point
        n_initial = min(20, budget // 10)
        best_x = None
        best_f = np.inf
        for _ in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: CMA-ES from the best starting point
        remaining_budget = budget - n_initial
        final_x, final_f, _ = self._cma_es(func, best_x, 2.0, remaining_budget, rng)
        
        return final_x, final_f
# EVOLVE-BLOCK-END
