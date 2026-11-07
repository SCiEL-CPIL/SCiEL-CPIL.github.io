"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        N = self.dim
        lambda_pop = 12
        mu = lambda_pop // 2
        
        # Initialization
        mean = np.zeros(N)
        sigma = 2.5
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation
        c_sigma = (mu_eff/2 + 1) / (N + 5)
        c_c = (4 + mu_eff/N) / (N + 4 + 2*mu_eff/N)
        c_1 = 2 / ((N + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((N + 2)**2 + mu_eff))
        
        pc = np.zeros(N)
        ps = np.zeros(N)
        C = np.eye(N)
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        while calls < self.budget:
            if time.time() > deadline - 5.0:
                break
            
            # Sample
            Y = np.random.randn(lambda_pop, N)
            
            # Compute C^0.5 components
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Z = (Y * D) @ B.T
            Z = (Y * D) @ B.T
            X = mean + sigma * Z
            
            F = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if time.time() > deadline - 5.0:
                    break
                F[i] = func(X[i])
                calls += 1
                
            # If we broke early, pad F with inf? 
            # If we broke, we should stop the outer loop.
            if time.time() > deadline - 5.0:
                break
                
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort
            order = np.argsort(F)
            X_parents = X[order[:mu]]
            
            # Update Mean
            m_old = mean
            mean = np.sum(w[:, None] * X_parents, axis=0)
            
            # Update Ps
            y = (mean - m_old) / sigma
            C_inv_sqrt = B @ np.diag(1.0/D) @ B.T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y)
            
            # Hessian for Sigma
            h_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lambda_pop))) < 
                     (1 + 2 / (N + 1)) * np.sqrt((N + 1) * c_sigma * (2 - c_sigma)))
            
            # Update Pc
            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # Update C
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_h * C)
            
            # Rank Mu Update
            Y_parents = (X_parents - m_old) / sigma
            C += c_mu * np.sum(w[:, None, None] * (Y_parents * Y_parents.T) * w, axis=0)
            
            # Recalculate Eigen
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            C = B @ np.diag(D**2) @ B.T
            
            # Update Sigma
            sigma *= np.exp(c_sigma / 5.0 * (np.linalg.norm(ps) / np.sqrt(N) - 1))
            
            if sigma < 1e-8 or sigma > 50:
                break

        # Polish Phase
        remaining = self.budget - calls
        if remaining > 0 and time.time() < deadline - 1.0:
            x = best_x
            fx = best_f
            sigma_polish = 0.1
            rng = np.random.default_rng(123)
            
            for _ in range(remaining):
                if time.time() > deadline - 1.0:
                    break
                y = x + sigma_polish * rng.standard_normal(N)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish *= 1.2
                else:
                    sigma_polish *= 0.8
                    
                if fx < best_f:
                    best_f = fx
                    best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
