import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        dim = self.dim
        budget = self.budget
        
        # Initial parameters
        # Start at center of the box
        mean = np.zeros(dim)
        sigma = 2.0 # Initial step size (box is [-5, 5], so 2 is reasonable)
        
        # Population size
        # Standard heuristic: lambda = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))
        if lam < 2: lam = 2
        
        # CMA-ES specific parameters
        # Learning rates
        mu = lam // 2  # Number of recombination weights
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Step size control parameters
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Covariance matrix adaptation parameters
        c1 = (2 / ((dim + 1.3)**2 + mu_eff)) * (2 - (dim + 1.3) / (dim + 1.3 + mu_eff))
        cmu = min(1 - c1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # Damping for the cumulative step size
        cc = (4 - mu_eff) / (dim + 4 - mu_eff)
        
        # Cumulation path
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        Cinv = np.eye(dim)
        D = 1.0
        B = np.eye(dim)
        
        # Precompute inverse square root of C
        def inv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            inv_sqrt_vals = 1.0 / np.sqrt(eigvals)
            return eigvecs @ np.diag(inv_sqrt_vals) @ eigvecs.T
            
        Cinv_sqrt = inv_sqrt(C)
        
        best_x = mean.copy()
        best_f = func(best_x)
        evals = 1
        
        # Main loop
        while evals < budget:
            # Generate offspring
            offspring = []
            for i in range(lam):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                x = mean + sigma * (Cinv_sqrt @ z)
                # Clip to box
                x = np.clip(x, -5, 5)
                offspring.append(x)
            
            # Evaluate offspring
            f_vals = np.array([func(x) for x in offspring])
            evals += lam
            
            # Update best
            idx_best = np.argmin(f_vals)
            if f_vals[idx_best] < best_f:
                best_f = f_vals[idx_best]
                best_x = offspring[idx_best].copy()
                
            # Sort offspring by fitness
            order = np.argsort(f_vals)
            y = np.array([offspring[i] for i in order])
            f_sorted = f_vals[order]
            
            # Recombination mean
            y_w = np.dot(weights[:mu], y[:mu])
            
            # Step size adaptation
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Cinv_sqrt @ y_w)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2) + 1 / (8 * dim**4))
            h_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**((evals - 1) * cs))) < (1.4 + 2 / (dim + 1)) * chi_n
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0) # Clip to prevent explosion or vanish
            
            # Update mean
            mean += sigma * y_w
            
            # Update covariance matrix
            # Update pc
            pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mu_eff) * y_w
            
            # Rank-mu update
            # h_c is the Heaviside step function for covariance update
            h_c = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c1)**((evals - 1) * c1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((1 - c1) / (1 + c1))
            
            # Update C
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - h_c + c1) * C) + cmu * sum(wi * (yi - mean_old) * (yi - mean_old).T)
            # Note: The standard update uses the shifted y values
            # y_shifted = y[:mu] - y_w
            # But the standard CMA-ES formula is:
            # C = (1 - c1 - cmu)C + c1(pc pc^T + (1 - h_c + c1)C) + cmu * sum(wi * (yi - mean) (yi - mean)^T)
            # Actually, the update is applied to the covariance matrix directly.
            
            # Let's use the standard update formula:
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - h_c + c1) * C)
            # + cmu * sum(wi * (yi - mean) @ (yi - mean).T)
            
            # Wait, the mean was updated *after* y_w calculation.
            # The vectors for the rank-mu update should be based on the old mean?
            # In standard CMA-ES:
            # 1. y_w is calculated from sorted y
            # 2. mean_new = mean_old + sigma * y_w
            # 3. C update uses (yi - mean_old) or (yi - mean_new)?
            # The standard formula uses the vectors relative to the *new* mean for the rank-mu part?
            # No, usually it's:
            # C = (1 - c1 - cmu)C + c1(pc pc^T + (1 - h_c + c1)C) + cmu * sum(wi * (yi - mean_new) (yi - mean_new)^T)
            # Let's verify. The paper says:
            # C_{t+1} = (1 - c1 - cmu)C_t + c1(pc_{t+1} pc_{t+1}^T + (1 - h_c + c1)C_t) + cmu * sum_{i=1}^mu w_i (y_{i:lambda} - y_w) (y_{i:lambda} - y_w)^T
            # Here y_{i:lambda} are the selected vectors, and y_w is the recombination mean.
            # So the vectors are relative to y_w, not the global mean.
            
            delta_c = c1 * (pc @ pc.T + (1 - h_c + c1) * C)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                # y[i] is the i-th best offspring
                # y_w is the weighted sum of the top mu offspring
                diff = y[i] - y_w
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 - c1 - cmu) * C + delta_c + cmu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Update Cinv_sqrt
            Cinv_sqrt = inv_sqrt(C)
            
            # Optional: Clip mean to box to ensure we stay within bounds for next generation?
            # CMA-ES handles boundaries by clipping samples, but the mean can drift.
            # It's safer to clip the mean to the box to prevent the distribution from centering outside.
            mean = np.clip(mean, -5, 5)

        return best_x, best_f
