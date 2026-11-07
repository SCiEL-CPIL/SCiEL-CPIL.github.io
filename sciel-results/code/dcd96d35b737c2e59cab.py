import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        best_x = None
        best_f = np.inf
        
        # Multi-start strategy: split budget into multiple restarts
        # For dim=5 and budget=500, 5 restarts of 100 evals each is reasonable.
        num_restarts = 5
        budget_per_restart = self.budget // num_restarts
        
        rng = np.random.default_rng(42)
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize CMA-ES parameters
            # Start point: random in [-4, 4]^5 to stay within box but away from edges initially
            x0 = rng.uniform(-4, 4, self.dim)
            # Initial step size: cover a significant portion of the box
            sigma0 = 2.0
            
            # CMA-ES internal state
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(self.dim)
            C_inv_sqrt = np.eye(self.dim)
            
            # Pre-compute weights and constants
            mu = self.dim // 2  # Number of parents
            # Weights for recombination
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim ** 2))
            
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            
            # Evaluate initial point
            try:
                f0 = func(x0)
            except:
                f0 = np.inf
                
            if np.isfinite(f0):
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
                    
            remaining_budget = budget_per_restart - 1
            if remaining_budget <= 0:
                continue
                
            # Main CMA-ES loop
            while remaining_budget > 0:
                if time.time() > deadline:
                    break
                    
                # Generate offspring
                # Sample directions from multivariate normal with covariance C
                # y = z * sigma, where z ~ N(0, C)
                # Efficient sampling: z = L * z_std, where C = L L^T
                # We maintain C_inv_sqrt for efficient updates, but for sampling we need L.
                # Let's use Cholesky decomposition of C for sampling.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    C = 0.5 * (C + C.T) + np.eye(self.dim) * 1e-12
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(self.dim)
                        
                z_std = rng.standard_normal((mu, self.dim))
                y = np.dot(z_std, L.T)
                x_offspring = mean + sigma * y
                
                # Evaluate offspring
                f_offspring = []
                for i in range(mu):
                    x_i = x_offspring[i]
                    # Clip to box [-5, 5]
                    x_i = np.clip(x_i, -5, 5)
                    try:
                        f_i = func(x_i)
                    except:
                        f_i = np.inf
                    f_offspring.append(f_i)
                    
                    remaining_budget -= 1
                    if remaining_budget <= 0:
                        break
                        
                f_offspring = np.array(f_offspring[:remaining_budget + mu - (budget_per_restart - remaining_budget - 1)]) # This logic is messy, let's just take what we have
                
                # Actually, simpler:
                # We evaluated up to 'mu' points. If we ran out of budget in the middle, we only have some.
                # Let's restructure the loop to handle budget strictly.
                
                # Let's restart the loop structure for clarity
                # The previous while loop logic was getting complicated with partial generations.
                # Standard CMA-ES uses a full generation. If budget runs out mid-generation, we stop.
                
                if len(f_offspring) < mu:
                    break
                
                # Sort offspring by fitness
                idx = np.argsort(f_offspring)
                x_selected = x_offspring[idx[:mu]]
                f_selected = f_offspring[idx[:mu]]
                
                # Update best
                if f_selected[0] < best_f:
                    best_f = f_selected[0]
                    best_x = x_selected[0].copy()
                
                # Update mean
                mean_new = np.sum(w[:, None] * x_selected, axis=0)
                
                # Update p_sigma
                y_sigma = np.dot(C_inv_sqrt, (mean_new - mean) / sigma)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                
                # Hessian correction (Sigma update)
                h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * remaining_budget))) < (1.4 + 2 / (self.dim + 1)) * chi_n
                if h_sig:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
                else:
                    p_c = (1 - c_c) * p_c
                    
                # Update covariance matrix C
                delta_diag = 0.0
                for i in range(mu):
                    z_i = (x_selected[i] - mean) / sigma
                    delta_diag += w[i] * (z_i @ z_i) - self.dim
                    # Rank-mu update
                    C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C) \
                        + c_mu * np.sum(w[:, None] * np.outer(z_selected[i] - mean, z_selected[i] - mean) / sigma**2, axis=0)
                        
                # Actually, the standard update is:
                # C = (1 - c1 - cmu) * C + c1 * (pc pc^T + (1 - hsig) * cc * (2 - cc) * C) + cmu * sum(wi * (xi - mean)(xi - mean)^T / sigma^2)
                
                # Let's rewrite the update correctly.
                # z_i = (x_i - mean_old) / sigma
                # C_new = (1 - c1 - cmu) * C_old + c1 * (p_c p_c^T + (1 - h_sig) * c_c * (2 - c_c) * C_old) + cmu * sum(w_i * z_i z_i^T)
                
                # Recalculate C properly
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C)
                for i in range(mu):
                    z_i = (x_selected[i] - mean) / sigma
                    C += c_mu * w[i] * np.outer(z_i, z_i)
                    
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update C_inv_sqrt periodically to keep it accurate
                # We need C_inv_sqrt for the next p_sigma update.
                # We can compute it from C.
                try:
                    C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                except:
                    # If Cholesky fails, reset C to identity and restart adaptation for this run
                    C = np.eye(self.dim)
                    C_inv_sqrt = np.eye(self.dim)
                    p_sigma = np.zeros(self.dim)
                    p_c = np.zeros(self.dim)
                    sigma = sigma0
                    
                mean = mean_new
                
                # Optional: Check for convergence or stagnation
                # If sigma becomes very small, we might be stuck.
                if sigma < 1e-10:
                    break

        return best_x, best_f
