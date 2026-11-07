import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        D = self.dim
        budget = self.budget
        
        call_count = [0]
        best_x = [np.zeros(D)]
        best_f = [np.inf]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count[0] >= budget:
                raise ValueError("Budget exceeded")
            
            call_count[0] += 1
            f_val = float(func(x))
            if f_val < best_f[0]:
                best_f[0] = f_val
                best_x[0] = x.copy()
            return f_val

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(D))
        # Initial standard deviation
        sigma = 2.0 
        # Initial mean
        m = np.zeros(D)
        # Covariance matrix
        C = np.eye(D)
        # Eigendecomposition cache
        B, D_eig = np.linalg.eigh(C)
        C_sqrt = B @ np.diag(np.sqrt(D_eig))
        # Step size control
        cs = (lam + 2) / (D + lam + 2)
        ds = 1 + 2 * max(0, np.sqrt((lam - 1) / (D + 1)) - 1) + cs
        # Path updates
        p_sigma = np.zeros(D)
        p_c = np.zeros(D)
        # Learning rates
        cmu = 2 / ((D + 1.3)**2 + lam)
        c_sigma = (lam + 2) / (D + 4)
        
        # Weights for recombination
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial budget allocation
        # CMA-ES is efficient, but we must ensure we don't waste time on setup.
        # With D=5, lam=14. 500 calls is about 35 generations.
        
        while call_count[0] < budget and time.time() < deadline:
            # Generate population
            pop_x = []
            pop_f = []
            
            for i in range(lam):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                z = np.random.randn(D)
                # Transform z using current C and sigma
                x_i = m + sigma * (C_sqrt @ z)
                # Clip to bounds manually to ensure valid input, though func clips too
                x_i = np.clip(x_i, -5.0, 5.0)
                try:
                    f_i = wrapped_func(x_i)
                    pop_x.append(x_i)
                    pop_f.append(f_i)
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    pass
            
            if len(pop_x) == 0:
                break
                
            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Update mean using weighted recombination of best mu individuals
            w = weights
            m_new = sum(w[i] * pop_x[i] for i in range(mu))
            
            # Update path variables
            # C^{-1/2} (m_new - m) / sigma
            # We need C_inv_sqrt. Since C = B D B^T, C_inv_sqrt = B D_inv_sqrt B^T
            D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D_eig, 1e-12))
            C_inv_sqrt = B @ np.diag(D_inv_sqrt) @ B.T
            
            c_inv_sqrt_m_diff = C_inv_sqrt @ (m_new - m) / sigma
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * c_inv_sqrt_m_diff
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mu_eff) * c_inv_sqrt_m_diff
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_lam = np.sqrt(lam) * (1 - 1 / (2 * lam**2) + 1 / (8 * lam**4)) # Approximation for chi
            # More accurate chi for standard normal: E[||N(0,I)||]
            # For D=5, chi ~ 2.13
            chi_D = np.sqrt(D) * (1 - 1 / (4 * D) + 7 / (32 * D**2))
            sigma *= np.exp(cs / ds * (norm_p_sigma / chi_D - 1))
            
            # Update covariance
            # h_sigma = (norm_p_sigma / sqrt(1 - (1 - cs)^{2*(call_count/lam)}) < 1.4 + 2/(D+1))
            # Simplified check: if norm is not too large
            h_sigma = 1.0 if norm_p_sigma / chi_D < 1.4 + 2.0 / (D + 1) else 0.0
            
            rank_mu = np.zeros((D, D))
            for i in range(mu):
                x_i = pop_x[i]
                diff = C_inv_sqrt @ (x_i - m) / sigma
                rank_mu += w[i] * np.outer(diff, diff)
                
            rank_one = np.outer(p_c, p_c)
            C = (1 + cmu * (h_sigma * (ds + cs - 1) - cs) + cmu * mu_eff) * C \
                - cmu * rank_mu \
                + cmu * (h_sigma * (ds + cs - 1) - cs) * rank_one
            
            # Ensure positive definiteness and update eigen decomposition
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Regularize if needed
            # Check condition number
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            B, D_eig = np.linalg.eigh(C)
            C_sqrt = B @ np.diag(np.sqrt(D_eig))
            
            m = m_new
            
            # Optional: If sigma becomes too small, we might be stuck, but CMA-ES handles this.
            # We just loop until budget is out.
            
        return (best_x[0], best_f[0])
