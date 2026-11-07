"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initial point: center of box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of parents
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Step size
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for step size
        D = np.sqrt(np.diag(C))

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Track history for AOCC (we just need to make calls)
        calls_made = 1

        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Transform using Cholesky-like (use eigen for simplicity, but for small dim just use diag approx)
                # Actually, let's do proper CMA-ES update
                # For efficiency and correctness with small dim, we use the standard update
                # But to keep it simple and robust, we approximate C update
                # Let's use a simpler covariance adaptation: rank-1 and rank-mu updates
                
                # Sample: x + sigma * B * z, where B is matrix square root of C
                # Use eigen decomposition each step for correctness (dim=5 is small)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                B = eigvecs @ np.diag(np.sqrt(eigvals))
                pop[i] = x + sigma * (B @ z)
                pop[i] = np.clip(pop[i], -5, 5)

            # Evaluate population
            fvals = np.array([func(xi) for xi in pop])
            calls_made += lambda_

            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = pop[idx_min].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Weighted mean of best mu
            y_w = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            x_new = x + sigma * y_w

            # Update step size
            y_sigma = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            # Actually y_sigma should be the weighted mean of the search steps, not positions
            # Let's redo: the step is (pop - x) / sigma
            steps = (pop_sorted[:mu] - x[None, :]) / sigma
            y_sigma = np.sum(w[:, None] * steps, axis=0)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            p_sigma_norm = np.linalg.norm(p_sigma)
            
            # Hessian-like correction
            h_sig = 1 if p_sigma_norm / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update covariance
            y_c = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c / sigma

            # Rank-1 update
            C = (1 + c_1 * (2 - h_sig) - c_1) * C + c_1 * np.outer(y_c, y_c)
            # Rank-mu update
            C += c_mu * np.sum(w**2 * (pop_sorted[:mu] - x[None, :])[:, None] * (pop_sorted[:mu] - x[None, :])[None, :], axis=0) / sigma**2

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
