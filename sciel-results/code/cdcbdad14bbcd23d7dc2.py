"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Allocate budget: 80% to CMA-ES, 20% to Nelder-Mead
        cma_budget = int(budget * 0.8)
        nm_budget = budget - cma_budget
        
        # --- Phase 1: CMA-ES ---
        # Parameters
        lambda_ = 4 * dim  # Population size
        mu = lambda_ // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) * (dim + mu_eff - 2)) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (32 * dim ** 2))
        
        # State variables
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = x_mean
        best_f = float('inf')
        
        # Main CMA-ES loop
        while calls < cma_budget:
            # Generate population
            z = rng.standard_normal((lambda_, dim))
            y = [x_mean + sigma * (B @ (D * z[i])) for i in range(lambda_)]
            
            # Evaluate population
            fvals = []
            for y_i in y:
                y_i_clipped = np.clip(y_i, -5, 5)
                f_i = func(y_i_clipped)
                fvals.append(f_i)
                calls += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = y_i_clipped.copy()
                
                # If we exceed budget during generation, break
                if calls >= cma_budget:
                    break
            
            # If we broke early, we might not have a full population to update
            # For simplicity, if we don't have at least mu+1 evaluations, we stop CMA-ES
            if len(fvals) < mu + 1:
                break
            
            # Sort by fitness
            idx = np.argsort(fvals)
            y_sorted = [y[i] for i in idx[:mu]]
            w = weights[:mu]
            
            # Update mean
            x_new = sum(w[i] * y_sorted[i] for i in range(mu))
            
            # Update step size control variable
            y_sigma = (x_new - x_mean) / (sigma * chi_N)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1))) < (1.4 + 2 / (dim + 1)) * chi_N
            if h_sigma:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = np.outer(p_c, p_c) if h_sigma else np.zeros((dim, dim))
            rank_one = sum(w[i] * np.outer(y_sorted[i] - x_mean, y_sorted[i] - x_mean) for i in range(mu))
            C = (1 + c_1 * (2 - c_1) - delta_h) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Recalculate eigenvalues (Cholesky decomposition)
            # Use SVD for stability
            U, s, Vt = np.linalg.svd(C)
            # Ensure positive definite
            s = np.maximum(s, 1e-6)
            C = (U * np.sqrt(s)) @ (U * np.sqrt(s)).T
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-6))
            
            # Update step size
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Update mean
            x_mean = x_new
            
            # Safety check: if sigma becomes too small or large, stop
            if sigma < 1e-8 or sigma > 1e6:
                break
        
        # --- Phase 2: Nelder-Mead ---
        # Use remaining budget for local refinement
        if nm_budget > 0 and calls < budget:
            # Create a wrapper to count calls and stop
            nm_calls = [0]
            def nm_func(x):
                if nm_calls[0] >= nm_budget:
                    return float('inf')
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                nm_calls[0] += 1
                return val
            
            # Start from best found by CMA-ES
            x0 = best_x
            res = minimize(nm_func, x0, method='Nelder-Mead', 
                           options={'maxiter': nm_budget, 'xatol': 1e-8, 'fatol': 1e-12})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5, 5)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
