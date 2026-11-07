import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        deadline = time.time() + 250.0  # Safety margin
        
        # IPOP-CMA-ES Parameters
        # Initial population size
        lam = 4 + 3 * int(np.floor(np.log10(D + 10))) 
        # For D=5, log10(15) ~ 1.17, floor is 1. lam = 4 + 3 = 7. Let's use a fixed small start like 6 or 8.
        # Standard CMA-ES often uses 4+3ln(D). For D=5, that's ~7.
        # Let's start with lam = 6.
        
        # IPOP parameters
        # Increase population if no improvement for 'inc' generations
        inc = 10 
        # Maximum population size
        lam_max = budget // 2  # Ensure we don't exceed budget in one iteration
        if lam_max < lam:
            lam_max = lam
            
        # Initial state
        x = np.zeros(D)
        sigma = 2.0
        C = np.eye(D)
        B = np.eye(D)
        D_inv = np.ones(D)
        
        # Evolution path
        p_sigma = np.zeros(D)
        p_c = np.zeros(D)
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (D + mu_eff + 5)
        c_c = (4 + mu_eff / D) / (D + 4 + 2 * mu_eff / D)
        c_1 = 2 / ((D + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((D + 2) ** 2 + mu_eff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (D + 1)) - 1) + c_sigma
        
        # Initial best
        best_x = x.copy()
        best_fx = func(best_x)
        calls_made = 1
        
        # Track stagnation for IPOP
        stagnation_counter = 0
        last_best_fx = best_fx
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Calculate Hessian matrix sqrt for sampling
            # C = B D^2 B^T
            # We need to update B and D_inv if C changes, but for simplicity in this loop
            # we can use eigendecomposition every iteration or maintain it.
            # For D=5, eigendecomposition is cheap.
            
            # Sample points
            z = np.random.randn(lam, D)
            # y = B @ (D_inv * z) ? No, y = B @ (z * D_inv) where D_inv = 1/sqrt(eigenvalues)
            # Actually, standard CMA-ES samples y = B @ (z * D_inv)
            # Let's compute eigendecomposition of C
            eigenvalues, B = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            D_inv = 1.0 / np.sqrt(eigenvalues)
            
            Y = z * D_inv[None, :]
            Y = B @ Y.T
            Y = Y.T
            
            # Clip to box
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls_made += lam
            
            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = Y_sorted[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # Check IPOP condition
            if stagnation_counter >= inc:
                # Double population size
                lam = min(lam * 2, lam_max)
                # Reset covariance to identity to broaden search
                C = np.eye(D)
                B = np.eye(D)
                D_inv = np.ones(D)
                p_sigma = np.zeros(D)
                p_c = np.zeros(D)
                # Reset stagnation
                stagnation_counter = 0
                # Optionally reset sigma if it's too small, but let's keep it
                # Or reset sigma to a reasonable value if it collapsed
                if sigma < 0.1:
                    sigma = 1.0
                    
            # Update CMA-ES parameters
            # Mean update
            w = weights
            x_new = np.sum(Y_sorted[:mu] * w[:, None], axis=0)
            
            # Path updates
            # p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mu_eff) * B^T * D * (x_new - x)/sigma
            # p_c = (1-c_c)p_c + sqrt(c_c(2-c_c)mu_eff) * B^T * D * (x_new - x)/sigma
            
            # We need B and D from current C (before update)
            # We already have B and D_inv from sampling. D = 1/D_inv.
            D = 1.0 / D_inv
            
            # Transform (x_new - x) to principal axes
            diff = (x_new - x) / sigma
            diff_trans = B.T @ (D * diff)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff_trans
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_trans
            
            # Sigma update
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lam)))) < (1.4 + 2 / (D + 1)) * np.sqrt((mu_eff + 1) / (D + 1))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)
            
            # Covariance update
            if h_sigma:
                delta_h = 1 - (1 - c_sigma) ** (calls_made - lam)
            else:
                delta_h = 0
                
            # Outer products of top mu solutions
            Y_m = Y_sorted[:mu]
            # Transform to principal axes
            Y_m_trans = (Y_m - x) / sigma
            Y_m_trans = B.T @ (D * Y_m_trans.T).T
            
            # Rank-1 update
            C += c_1 * (1 + (1 - h_sigma) * c_sigma / (1 - c_sigma) * mu_eff - 1) * np.outer(p_c, p_c) - (c_1 / (1 - c_1)) * C
            
            # Rank-mu update
            # Sum of outer products of weighted differences
            rank_mu = np.sum(w[:, None, None] * np.outer(Y_m_trans.T, Y_m_trans.T), axis=0)
            C += c_mu * (rank_mu - (1 - h_sigma) * delta_h * C)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Move mean
            x = x_new
            
            # If population size grew, we might have used up budget
            if calls_made >= budget:
                break

        return best_x, best_fx
