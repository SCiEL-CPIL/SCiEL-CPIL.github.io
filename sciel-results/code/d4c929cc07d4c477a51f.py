import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Use a safety margin well below the 330s hard limit
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For dim=5, lambda=10 is reasonable, maybe slightly larger for exploration
        # Standard suggests lambda = 4 + floor(3*log(dim)) = 4 + floor(3*1.6) = 4+4=8. 
        # Or lambda = 2*dim = 10. Let's stick with 10.
        lam = 10
        mu = lam // 2  # Number of parents
        
        # Track best found during CMA-ES for polish
        best_x_cma = None
        best_f_cma = np.inf
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Initial parameters
        # The task states the minimizer is in [-4, 4]^5. 
        # A good initial sigma is roughly half the domain width to cover the space initially.
        # Domain is [-5, 5], width 10. Minimizer in [-4, 4].
        # Let's start with sigma = 2.0 and mean at 0.
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix
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
        
        # Pre-compute Cholesky factors initially
        L = np.linalg.cholesky(C)
        
        # Helper to update Cholesky
        def update_cholesky(C_new):
            try:
                L_new = np.linalg.cholesky(C_new)
                return L_new, True
            except np.linalg.LinAlgError:
                # Regularize if not PD
                C_reg = C_new + 1e-4 * np.eye(dim)
                try:
                    L_new = np.linalg.cholesky(C_reg)
                    return L_new, False
                except:
                    return L, False

        # Evaluation loop
        # We need to be careful with the number of calls.
        
        while calls_used + lam <= budget:
            if time.time() > deadline:
                break
                
            # Update L from C
            # In standard CMA-ES, we update C, then decompose.
            # But for sampling, we need L such that C = L L^T.
            # We can just decompose C at the start of each generation.
            L, _ = update_cholesky(C)
            
            # Generate population
            # z ~ N(0, I)
            z = np.random.randn(lam, dim)
            # y = z @ L.T
            y = z @ L.T
            x_pop = x_mean + sigma * y
            
            # Clip to box [-5, 5]
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.array([func(x) for x in x_pop])
            calls_used += lam
            
            # Update best
            idx_best = np.argmin(f_pop)
            if f_pop[idx_best] < best_f:
                best_f = f_pop[idx_best]
                best_x = x_pop[idx_best].copy()
                best_x_cma = x_pop[idx_best].copy()
                best_f_cma = f_pop[idx_best]
                
            # Sort
            idx_sort = np.argsort(f_pop)
            x_sorted = x_pop[idx_sort]
            f_sorted = f_pop[idx_sort]
            
            # Selection
            x_w = x_sorted[:mu]
            y_w = y[idx_sort][:mu] # Need the y vectors for the selected ones
            
            # Update mean
            x_mean_old = x_mean
            x_mean = np.sum(weights[:, np.newaxis] * x_w, axis=0)
            
            # Update p_sigma
            # d_mean = x_mean - x_mean_old
            d_mean = x_mean - x_mean_old
            
            # C^{-1/2} update approximation
            # We use L_inv_T = L^{-T}
            try:
                L_inv_T = np.linalg.inv(L).T
            except:
                L_inv_T = np.eye(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L_inv_T @ (d_mean / sigma))
            
            # h_sigma
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim**2))
            # Approximation for the norm check
            gen = calls_used / lam
            # c_hsig threshold
            c_hsig = (1 + 2 / max(1, mueff - 1) + c_sigma) / (1 + c_sigma)
            
            if gen > 0:
                # Proper check requires sum of (1-c_sigma)^{2t}
                # Approximation:
                norm_ps = np.linalg.norm(p_sigma)
                denom = np.sqrt(1 - (1 - c_sigma) ** (2 / gen))
                h_sigma = 1.0 if (norm_ps / denom) < (1.4 + 2 / (dim + 1)) * c_hsig else 0.0
            else:
                h_sigma = 1.0
                
            # Update sigma
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (L_inv_T @ (d_mean / sigma))
            
            # Update C
            # Rank-one update
            # delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_h if h_sigma < 0.5 else 1 - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                + c_mu * np.sum(weights[:, np.newaxis]**2 * (y_w - np.mean(y_w, axis=0))[:, np.newaxis] * (y_w - np.mean(y_w, axis=0)), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # If time is running out, try to evaluate the best point again? No, just break.
            
        # Final high-precision polish pass
        # Use remaining budget for fine-grained local search around the best point
        if best_x is not None and calls_used < budget and time.time() < deadline:
            polish_sigma = 0.01
            polish_budget = budget - calls_used
            polish_calls = 0
            polish_deadline = deadline - 5.0  # extra safety margin
            last_improvement_call = calls_used
            
            while polish_calls < polish_budget and time.time() < polish_deadline:
                # Generate a small neighborhood around the best point
                n_polish = min(10, polish_budget - polish_calls)
                
                # Try different step sizes: start small, increase if stuck
                candidates = []
                
                # Always include the current best
                candidates.append(best_x.copy())
                
                # Random perturbations
                for i in range(min(5, n_polish - 1)):
                    perturbation = np.random.randn(dim) * polish_sigma
                    candidate = np.clip(best_x + perturbation, -5, 5)
                    candidates.append(candidate)
                
                # Axis-aligned steps
                for axis in range(dim):
                    if len(candidates) >= n_polish:
                        break
                    step = np.zeros(dim)
                    step[axis] = polish_sigma
                    candidates.append(np.clip(best_x + step, -5, 5))
                    candidates.append(np.clip(best_x - step, -5, 5))
                
                # Evaluate candidates
                improved = False
                for cand in candidates[:n_polish]:
                    if calls_used >= budget or time.time() >= polish_deadline:
                        break
                    f_val = func(cand)
                    calls_used += 1
                    polish_calls += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = cand.copy()
                        polish_sigma = max(polish_sigma * 0.5, 1e-8)  # shrink on improvement
                        improved = True
                        last_improvement_call = calls_used
                
                if polish_calls >= polish_budget:
                    break
                
                # If no improvement in last 20 calls, slightly increase sigma
                if not improved and (calls_used - last_improvement_call) >= 20:
                    polish_sigma = min(polish_sigma * 1.5, 0.05)
        
        return best_x, best_f
