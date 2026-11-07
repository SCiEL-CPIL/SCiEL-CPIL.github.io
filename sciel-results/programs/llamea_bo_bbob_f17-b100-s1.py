"""CMA-ES + Nelder-Mead hybrid optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using a hybrid CMA-ES and Nelder-Mead strategy.
    1. CMA-ES for global search and handling rotations/shifts.
    2. Nelder-Mead for local refinement and fast convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            # Clip to valid range
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            
            try:
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return float(fx)
            except Exception:
                calls += 1
                return np.inf

        # --- Phase 1: CMA-ES ---
        # Parameters
        # lambda_ = 4 + floor(3 * log(n))
        lambda_ = 10  # Slightly larger than standard for n=5 to ensure exploration
        mu = max(1, int(lambda_ / 2))
        
        # Use standard CMA-ES weight formula for better convergence
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Standard adaptation rates
        c_sigma = (1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + mu_eff / (n + 1.0)) / (n + 4.0 + 2.0 * mu_eff / n)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu / n)
        c_mu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0)**2 + mu_eff)
        
        # Paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Initial values
        mean = np.zeros(n)
        C = np.eye(n)
        sigma = 2.0  # Large initial step to cover [-5, 5]
        
        # Adaptation rates
        c_sigma = (1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + mu_eff / (n + 1.0)) / (n + 4.0 + 2.0 * mu_eff / n)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu / n)
        c_mu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0)**2 + mu_eff)
        
        # Paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Initial evaluation
        f0 = evaluate(mean)
        
        # Main CMA-ES Loop
        # We reserve about 15% of the budget for local search
        cma_budget = int(budget * 0.85)
        
        while calls < cma_budget and time.time() < deadline:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
                
            Z = np.random.randn(lambda_, n)
            X = mean + sigma * (Z @ L.T)
            
            # Evaluate
            F = np.array([evaluate(x) for x in X])
            
            if calls >= cma_budget:
                break
                
            # Sort
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            if np.isinf(F_sorted[0]):
                break
                
            # Update mean
            X_top = X_sorted[:mu]
            mean_new = np.sum(w[:, np.newaxis] * X_top, axis=0)
            
            # Standardize
            v = (mean_new - mean) / sigma
            try:
                y_p = np.linalg.solve(L, v)
            except np.linalg.LinAlgError:
                y_p = v
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * y_p
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            E_norm = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 7.0/(32.0*n**2))
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / E_norm - 1.0))
            sigma = np.clip(sigma, 1e-12, 5.0)
            
            # Update p_c
            # C^-1/2 v
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                y_c = C_inv_half @ v
            except np.linalg.LinAlgError:
                y_c = v
                
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * y_c
            
            # Update C
            c_mu_adapt = c_mu * (1.0 - c_sigma / d_sigma * (norm_p_sigma**2 - 1.0))
            c_mu_adapt = np.clip(c_mu_adapt, 0.0, 2.0 / mu_eff)
            
            Y_top = (X_top - mean) / sigma
            try:
                Y_std = np.linalg.solve(L, Y_top.T).T
            except np.linalg.LinAlgError:
                Y_std = Y_top
                
            outer_sum = Y_std.T @ (w[:, np.newaxis] * Y_std)
            C = (1 - c_mu_adapt) * C + c_mu_adapt * outer_sum
            C = (C + C.T) / 2.0
            
            mean = mean_new
            
            # Convergence check
            if sigma < 1e-8 or (norm_p_sigma < 1e-8 and np.linalg.norm(mean_new - mean) < 1e-8):
                break
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Initial values
        mean = np.zeros(n)
        C = np.eye(n)
        sigma = 2.0  # Large initial step to cover [-5, 5]
        
        # Adaptation rates
        c_sigma = (1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + mu_eff / (n + 1.0)) / (n + 4.0 + 2.0 * mu_eff / n)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu / n)
        c_mu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0)**2 + mu_eff)
        
        # Paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Initial evaluation
        f0 = evaluate(mean)
        
        # Main CMA-ES Loop
        # We reserve about 10% of the budget for local search
        cma_budget = int(budget * 0.85)
        
        while calls < cma_budget and time.time() < deadline:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
                
            Z = np.random.randn(lambda_, n)
            X = mean + sigma * (Z @ L.T)
            
            # Evaluate
            F = np.array([evaluate(x) for x in X])
            
            if calls >= cma_budget:
                break
                
            # Sort
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            if np.isinf(F_sorted[0]):
                break
                
            # Update mean
            X_top = X_sorted[:mu]
            mean_new = np.sum(w[:, np.newaxis] * X_top, axis=0)
            
            # Standardize
            v = (mean_new - mean) / sigma
            try:
                y_p = np.linalg.solve(L, v)
            except np.linalg.LinAlgError:
                y_p = v
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * y_p
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            E_norm = np.sqrt(2.0 * (n + 1.0)) * np.exp(0.5 * (np.log((n + 1.0) / 2.0) - 1.0))
            # Approximation for E||N(0,I)||
            E_norm = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 7.0/(32.0*n**2))
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / E_norm - 1.0))
            sigma = np.clip(sigma, 1e-12, 5.0)
            
            # Update p_c
            # C^-1/2 v
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                y_c = C_inv_half @ v
            except np.linalg.LinAlgError:
                y_c = v
                
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * y_c
            
            # Update C
            c_mu_adapt = c_mu * (1.0 - c_sigma / d_sigma * (norm_p_sigma**2 - 1.0))
            c_mu_adapt = np.clip(c_mu_adapt, 0.0, 2.0 / mu_eff)
            
            Y_top = (X_top - mean) / sigma
            try:
                Y_std = np.linalg.solve(L, Y_top.T).T
            except np.linalg.LinAlgError:
                Y_std = Y_top
                
            outer_sum = Y_std.T @ (w[:, np.newaxis] * Y_std)
            C = (1 - c_mu_adapt) * C + c_mu_adapt * outer_sum
            C = (C + C.T) / 2.0
            
            mean = mean_new
            
            # Convergence check
            if sigma < 1e-8 or (norm_p_sigma < 1e-8 and np.linalg.norm(mean_new - mean) < 1e-8):
                break

        # --- Phase 2: Nelder-Mead Refinement ---
        # If we still have budget and found a finite best point
        if not np.isinf(best_f) and calls < budget and time.time() < deadline:
            try:
                # Use remaining budget roughly as maxiter
                # Nelder-Mead maxiter is number of function evaluations? No, it's iterations.
                # In scipy, maxiter is max number of iterations.
                # For safety, limit iterations to remaining budget / n (approx)
                max_iter = max(10, int((budget - calls) / max(1, n)))
                
                res = minimize(
                    lambda x: evaluate(x), 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
