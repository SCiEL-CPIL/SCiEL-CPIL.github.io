"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s kill

        def safe_call(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return np.inf

        # Global bookkeeping
        best_x = None
        best_f = np.inf
        calls_made = 0

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2  # Number of parents
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size update rate
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_h = (5 + mu_eff / dim) / (dim + 9)
        
        # Initial mean and step size
        m = np.zeros(dim)
        sigma = 2.0
        
        # Initial covariance
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper for matrix square root/inverse via eigendecomposition
        def update_matrices():
            nonlocal C, C_inv, C_sqrt_inv, C_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.maximum(eigvals, 1e-16)
            C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            C_inv = (eigvecs * (1.0 / eigvals)) @ eigvecs.T
            C_sqrt_inv = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

        update_matrices()

        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            xs = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                x = m + sigma * y
                ys.append(y)
                xs.append(x)
            
            # Evaluate population
            fs = []
            for x in xs:
                fx = safe_call(x)
                fs.append(fx)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Sort by fitness
            sorted_indices = np.argsort(fs)
            ys = [ys[i] for i in sorted_indices]
            xs = [xs[i] for i in sorted_indices]
            fs = [fs[i] for i in sorted_indices]
            
            # Update mean
            m_old = m
            m = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update sigma
            y_mean = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ y_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance
            y_diff = m - m_old
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_sqrt_inv @ y_diff / sigma
            
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff) else 0
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys[i], ys[i])
            
            rank_one = np.outer(p_c, p_c)
            
            C = (1 + c_c * (h_sigma * (1 / c_c - 1) - 1) - c_c) * C + c_c * rank_one + c_c * (1 - h_sigma) * rank_mu
            
            # Ensure positive definiteness by clipping eigenvalues if necessary
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-16:
                eigvals = np.maximum(eigvals, 1e-16)
                C = (eigvecs * eigvals) @ eigvecs.T
            
            update_matrices()
            
            # Stop if sigma is too small or no improvement
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
