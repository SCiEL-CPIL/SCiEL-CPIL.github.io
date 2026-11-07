"""
Rewritten Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
CMA-ES is chosen for its robustness to rotation and scaling, which are present in the hidden transformation.
It adapts the search distribution to the problem's local geometry, leading to faster convergence on Rastrigin.
A time guard ensures the program finishes within the 300s limit.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        
        # Weights
        mu = lam // 2
        # Logarithmic weights
        weights = np.log(lam / 2 + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff / 2 + 1) / (n + 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start at the center of the box
        m = np.zeros(n)
        sigma = 1.0 # Initial step size, covers a good portion of [-5,5]
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute Cholesky factor
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(n)
            
        best_x = m.copy()
        best_f = func(best_x)
        
        start_time = time.time()
        deadline = start_time + 280.0 # Safety margin for 300s limit
        
        calls_made = 1
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for _ in range(lam):
                z = np.random.randn(n)
                y = m + sigma * (C_sqrt @ z)
                # Clip to box to ensure valid input (though func clips too, 
                # doing it here avoids potential issues if func expects strict bounds)
                y = np.clip(y, -5.0, 5.0)
                ys.append(y)
            
            # Evaluate population
            fs = []
            for y in ys:
                f_val = func(y)
                fs.append(f_val)
                calls_made += 1
                if time.time() > deadline:
                    break
            
            # If we broke out of the loop early due to time, pad fs if necessary
            # But since we break the while loop, we just exit.
            # However, we need to ensure we don't crash if fs is shorter than lam.
            if len(fs) < lam:
                # Fill with worst value to maintain array shapes if we update, 
                # but actually we should just stop updating if we don't have full population.
                # For simplicity and safety, if we run out of time, we break the main loop.
                break
            
            fs = np.array(fs)
            ys = np.array(ys)
            
            # Update best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = ys[idx_best]
                
            # Sort population by fitness
            sort_idx = np.argsort(fs)
            ys_sorted = ys[sort_idx]
            fs_sorted = fs[sort_idx]
            
            # Update Mean
            m_new = m + sigma * (weights @ ys_sorted[:mu])
            
            # Update Evolution Path for Sigma (ps)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ ((m_new - m) / sigma))
            
            # Update Sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update Evolution Path for Covariance (pc)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt.T @ ((m_new - m) / sigma))
            
            # Update Covariance
            delta_h = int((1 + 0.3 * (mu_eff - n) / (n + mu_eff)) * np.linalg.norm(pc)**2 / n < (1 - c_c)**2)
            
            # Rank-one update
            C = (1 + c_1 * delta_h * (1 - c_c) - c_1) * C + c_1 * np.outer(pc, pc)
            
            # Rank-mu update
            rank_mu = 0
            for i in range(mu):
                y_i = ys_sorted[i]
                rank_mu += weights[i] * np.outer(y_i, y_i)
            C += c_mu * rank_mu
            
            # Ensure positive definiteness
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small diagonal if Cholesky fails
                C += 1e-12 * np.eye(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(n)
                    
            m = m_new
            
            # Optional: If sigma is too small, we might be stuck. 
            # But CMA-ES handles this well. 
            # If sigma is too large, it explores.
            
        return best_x, best_f
