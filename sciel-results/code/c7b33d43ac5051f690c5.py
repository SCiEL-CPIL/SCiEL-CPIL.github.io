"""CMA-ES optimizer with deadline guard for BBOB f24 (Lunacek bi-Rastrigin).

Strategy:
  1. Start from the center of the box (0,0,0,0,0) which is a safe initial guess
     given the hidden shift is bounded within [-4, 4]^5.
  2. Use a large initial sigma (2.0) to allow global exploration of the box.
  3. Let CMA-ES adaptively reduce sigma and adjust the covariance matrix to
     perform local refinement.
  4. Use the full budget (500) for the CMA-ES loop, avoiding the waste of
     separate grid search phases which proved suboptimal in 5D.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard for 5D box-constrained problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills after 330s. We return well before that.
        deadline = start_time + 320.0

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        # Population size: lambda = 4 + 3 * ln(n)
        # For n=5: 4 + 3*1.6 = 8.8 -> 9. 
        # Standard recommendation is often 2*n+1 = 11 for small n.
        # Let's use 10 or 11. Inspiration 1 used 11.
        lam = max(4 + int(3 * np.log(n)), 2 * n + 1)
        lam = min(lam, budget // 2)  # Don't use more than half budget for one generation
        
        # Initial point: center of box
        x = np.zeros(n)
        fx = func(x)
        evals = 1
        
        # Initial sigma: 2.0 allows exploration of the [-5, 5] box
        # (approx 2.5 sigma range covers +/- 5)
        sigma = 2.0
        
        # Covariance matrix C = I
        C = np.eye(n)
        # Inverse of C
        C_inv = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Parameters
        mu = lam // 2  # Number of selected individuals
        # Logarithmic weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
        
        # Track best
        best_x = x.copy()
        best_f = fx
        
        # Main loop
        while evals < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate offspring
            ys = []
            fvals = []
            for i in range(lam):
                if evals >= budget:
                    break
                # Sample from multivariate normal with covariance C
                # y = x + sigma * z where z ~ N(0, C)
                z = rng.multivariate_normal(np.zeros(n), C, size=1).flatten()
                y = x + sigma * z
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                ys.append(y)
                fvals.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(ys) == 0:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            # Select top mu
            selected_indices = order[:mu]
            ys_sel = [ys[i] for i in selected_indices]
            fvals_sel = [fvals[i] for i in selected_indices]
            
            # Weighted mean of selected
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * ys_sel[i]
            
            # Update evolution paths
            # p_sigma
            delta_x = x_new - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv @ (delta_x / sigma)
            
            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma) ** (evals + 1))
            h_sigma = 1 if (norm_p_sigma / denom) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv @ (delta_x / sigma)
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 + delta_h) * C
            C -= c_1 * (np.outer(p_c, p_c) + delta_h * C)
            for i in range(mu):
                y_i = delta_x / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(n)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x = x_new
            C_inv = np.linalg.inv(C)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
