"""Optimizer for BBOB-style f15 (Rastrigin rotated) with Multi-start CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility within the run
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def time_left():
            return time.time() < deadline and calls_made < budget

        # CMA-ES Parameters
        # lambda = 4 + 3*log(dim)
        lam = int(4 + 3 * np.log(dim))
        if lam < 4: lam = 4
        
        # Multi-start strategy
        # We will run several independent CMA-ES instances.
        # With budget 500 and lam=7, each instance can run for ~100-200 evals.
        # Let's try 4 restarts of ~100 evals each, plus a final polish.
        
        n_restarts = 4
        budget_per_restart = budget // (n_restarts + 1) # Leave some for polish

        for r in range(n_restarts):
            if not time_left():
                break
            
            # Initialize CMA-ES state
            # Random start point in [-5, 5]
            x_mean = rng.uniform(-5, 5, dim)
            sigma = 2.0 # Initial step size
            
            # Covariance matrix C = I
            C = np.eye(dim)
            # Inverse square root of C for sampling
            # Initially C=I, so C^-0.5 = I
            C_inv_sqrt = np.eye(dim)
            
            # Step size adaptation parameters
            mu = lam // 2 # Number of recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 - mu_eff + nu) / ((dim + 2) * (mu_eff + 4 - nu)) if False else (4 - mu_eff + 2) / ((dim + 2) * (mu_eff + 4 - 2)) # Simplified, standard c_c
            # Standard c_c = (4 - mu_eff + nu) / ((dim + 2)(mu_eff + 4 - nu)) where nu = 2/(dim+1.3)
            nu = 2.0 / (dim + 1.3)
            c_c = (4 - mu_eff + nu) / ((dim + 2) * (mu_eff + 4 - nu))
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = (min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)) / (1 - c_1))
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Initial evaluation
            fx = func(x_mean)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_mean.copy()
            
            # Main CMA-ES loop
            # We run until budget_per_restart is used or time runs out
            # Each generation consumes lam evaluations
            while time_left() and calls_made + lam <= budget:
                # 1. Sample lambda individuals
                # y = C_inv_sqrt @ z
                Z = rng.standard_normal((lam, dim))
                Y = Z @ C_inv_sqrt.T # Shape (lam, dim)
                X = x_mean[None, :] + sigma * Y # Shape (lam, dim)
                
                # Clip to box
                X = np.clip(X, -5, 5)
                
                # 2. Evaluate
                F = np.array([func(xi) for xi in X])
                calls_made += lam
                
                # Update global best
                min_idx = np.argmin(F)
                if F[min_idx] < best_f:
                    best_f = F[min_idx]
                    best_x = X[min_idx].copy()
                
                # 3. Recombine
                # Sort indices by fitness
                idx = np.argsort(F)
                X_sel = X[idx[:mu]]
                F_sel = F[idx[:mu]]
                
                # New mean
                x_mean_new = np.sum(weights[:, None] * X_sel, axis=0)
                
                # 4. Adaptation of Covariance Matrix and Step Size
                # Update evolution paths
                # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mu_eff) * C_inv_sqrt * (x_mean_new - x_mean)/sigma
                # p_c = (1 - c_c) p_c + sqrt(c_c(2-c_c)mu_eff) * (x_mean_new - x_mean)/sigma
                
                # Note: We need to be careful with the update order.
                # Standard CMA-ES updates p_sigma and p_c using the difference of means.
                
                # Calculate delta
                delta = (x_mean_new - x_mean) / sigma
                
                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ delta
                
                # Update p_c
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta
                
                # Check if p_sigma is too large (hysteresis for sigma adaptation)
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
                if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (dim + 1)) * (1 - c_sigma):
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                    
                # Update Covariance Matrix
                # H = (X_sel - x_mean_new).T @ (X_sel - x_mean_new) / sigma^2
                # Actually, we use the weighted sum of outer products
                # DeltaX = (X_sel - x_mean) / sigma
                DeltaX = (X_sel - x_mean) / sigma
                H = np.sum(weights[:, None, None] * (DeltaX[:, :, None] * DeltaX[:, None, :]), axis=0)
                
                # Update C
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) C) + c_mu H
                # h_c is 1 if ||p_c||/sqrt(1-(1-c_c)^{2gen}) < (1.4 + 2/(dim+1)) * chi_n, else 0
                # We approximate h_c = 1 for simplicity or use a static threshold
                # For robustness with small budgets, let's use a simplified update
                # C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * C) + c_mu * H
                # Wait, the standard formula is:
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) C) + c_mu * sum(wi (xi - xmean_new)(xi - xmean_new)^T)
                
                # Let's use the standard update without h_c for simplicity (h_c=1)
                # This is safer for small populations
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + C) + c_mu * H
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Check PD
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                        C += 1e-10 * np.eye(dim)
                except:
                    C = np.eye(dim)
                    
                # Update C_inv_sqrt
                # Use eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                
                x_mean = x_mean_new
                
                # Reset p_c and p_sigma if sigma becomes too small or large?
                # Standard CMA-ES doesn't reset, but we can stop if sigma is too small
                if sigma < 1e-8:
                    break

        # Final Polish
        if best_x is not None and time_left():
            x_mean = best_x.copy()
            fx = best_f
            sigma = 0.5 # Smaller sigma
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            
            # Small CMA-ES or (1+1) ES polish
            # Let's do a simple (1+1) ES polish with adaptive sigma
            sigma = 0.5
            while time_left():
                y = x_mean + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x_mean = y
                    fx = fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x_mean.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-5:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
