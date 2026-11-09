import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        C = np.eye(dim)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 - mueff / (dim + 4)) / (dim + 4 + mueff)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def track_call(x):
            nonlocal calls_used, best_x, best_f
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim**2))
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            z = np.random.randn(lam, dim)
            
            # Get Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-4 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
            
            # Sample
            y = z @ L.T
            x_pop = x_mean + sigma * y
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.array([track_call(x) for x in x_pop])
            
            # Check if we've exceeded budget
            if calls_used >= budget:
                break
            
            # Sort
            idx_sort = np.argsort(f_pop)
            x_sorted = x_pop[idx_sort]
            f_sorted = f_pop[idx_sort]
            z_sorted = z[idx_sort]
            
            # Selection
            x_w = x_sorted[:mu]
            z_w = z_sorted[:mu]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * x_w, axis=0)
            
            # Compute C^{-1/2} = L^{-T}
            try:
                L_inv_T = np.linalg.inv(L).T
            except np.linalg.LinAlgError:
                L_inv_T = np.eye(dim)
            
            d_mean = x_mean - x_mean_old
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L_inv_T @ (d_mean / sigma))
            
            # h_sigma check
            gen = max(1, calls_used / lam)
            c_hsig = (1 + 2 / max(1, mueff - 1) + c_sigma) / (1 + c_sigma)
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(max(1e-10, 1 - (1 - c_sigma) ** (2 / gen)))
            h_sigma = 1.0 if (norm_p_sigma / denom) < (1.4 + 2 / (dim + 1)) * c_hsig else 0.0
            
            # Update sigma
            sigma *= np.exp(c_sigma * (norm_p_sigma / chi_N - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (L_inv_T @ (d_mean / sigma))
            
            # Update C
            # Standardized variables for selected parents
            v_w = np.array([L_inv_T @ (x_w[i] - x_mean_old) / sigma for i in range(mu)])
            
            rank_mu = np.sum(weights[:, np.newaxis] * np.outer(v_w.T, v_w.T), axis=0)
            rank_1 = np.outer(p_c, p_c)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_1 + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Final polish pass: use remaining budget for fine-grained local search
        if best_x is not None and calls_used < budget:
            polish_sigma = 0.01
            polish_deadline = min(deadline, start_time + 285.0)
            while calls_used < budget and time.time() < polish_deadline:
                # Generate a few candidates around best_x
                n_polish = min(5, budget - calls_used)
                if n_polish <= 0:
                    break
                z_polish = np.random.randn(n_polish, dim)
                x_polish = best_x + polish_sigma * z_polish
                x_polish = np.clip(x_polish, -5, 5)
                
                improved = False
                for x in x_polish:
                    if calls_used >= budget or time.time() > polish_deadline:
                        break
                    val = track_call(x)
                    if val < best_f:
                        improved = True
                        best_x = x.copy()
                        best_f = val
                
                if improved:
                    polish_sigma *= 0.5
                else:
                    polish_sigma *= 0.5
                    if polish_sigma < 1e-4:
                        break
        
        return best_x, best_f
