"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES variant with population-based search for rotated/anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        # Weights for the mu best individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        # Step size update parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        def cholesky_inv_sqrt(C):
            """Compute inverse square root of covariance matrix C."""
            # Use eigenvalue decomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            C_inv_sqrt = (eigvecs * eigvals_inv_sqrt) @ eigvecs.T
            return C_inv_sqrt
        
        def update_chol(C):
            """Update Cholesky factor and inverse square root."""
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_inv_sqrt = cholesky_inv_sqrt(C)
            except np.linalg.LinAlgError:
                # Fallback to eigenvalue decomposition
                C_inv_sqrt = cholesky_inv_sqrt(C)
                # Re-regularize C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = (eigvecs * eigvals) @ eigvecs.T
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt = L
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)
            return C, C_inv_sqrt, C_sqrt
        
        C, C_inv_sqrt, C_sqrt = update_chol(C)
        
        # Initial evaluations
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Initial population
        for i in range(lam):
            z = rng.standard_normal(dim)
            x_i = x_mean + sigma * (C_sqrt @ z)
            fx_i = evaluate(x_i)
            if time.time() > deadline:
                break
        
        if calls >= budget or time.time() > deadline:
            if best_x is not None:
                return best_x, best_f
            return np.zeros(dim), float('inf')
        
        # Main loop
        while calls < budget and time.time() < deadline:
            # Generate population
            pop = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (C_sqrt @ z)
                pop.append(x_i)
            
            if len(pop) == 0:
                break
                
            # Evaluate population
            pop_fx = []
            for x_i in pop:
                fx_i = evaluate(x_i)
                pop_fx.append(fx_i)
                if calls >= budget or time.time() > deadline:
                    break
            
            if len(pop_fx) < mu:
                break
            
            # Sort by fitness
            indices = np.argsort(pop_fx)
            y = np.array([pop[i] for i in indices[:mu]])
            
            # Update mean
            x_new_mean = np.sum(w[:, None] * y, axis=0)
            
            # Update step size
            y_sigma = (y - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ x_new_mean - x_mean) / sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update covariance matrix
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) / chi_n < 1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + hsig * c_c * (x_new_mean - x_mean) / sigma
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_hsig) * C \
                + c_1 * (np.outer(pc, pc) - (1 - c_c) * (1 - 2 * c_c) * C) \
                + c_mu * np.sum(w[:, None] * np.outer(y - x_mean, y - x_mean) / sigma**2, axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            C, C_inv_sqrt, C_sqrt = update_chol(C)
            
            x_mean = x_new_mean
            
            # Check for convergence
            if np.linalg.norm(C_sqrt) < 1e-12:
                break
        
        if best_x is not None:
            return best_x, best_f
        return np.zeros(dim), float('inf')
# EVOLVE-BLOCK-END
