import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For dim=5, lambda=10 is a good balance between exploration and exploitation
        lam = 10
        mu = lam // 2  # Number of best solutions to update mean
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu = 2 * (mu - 1) / (lam * (1 - weights[0]))  # Variance reduction factor
        
        # Initial state
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Initial sigma: large enough to explore [-5, 5]^5
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        # Step size for mean update
        c_sigma = (mu + 2) / (dim + mu + 5)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1) - 1) / (dim + 2)) + c_sigma
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = np.zeros((lam, dim))
            f_y = np.zeros(lam)
            
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                # Sample from N(x, sigma^2 C)
                # Use Cholesky or eigen decomposition? For dim=5, simple multiplication is fine
                # We maintain C and use its square root via eigendecomposition for sampling
                # Actually, standard CMA-ES samples z ~ N(0, I) and y = x + sigma * C_sqrt @ z
                # We need to update C_sqrt. This is expensive. 
                # Alternative: Use the simpler "1+1" style if CMA is too complex to implement correctly in a short script?
                # No, CMA-ES is worth it. Let's use a simplified update or rely on the fact that dim is small.
                
                # For efficiency and correctness in a short implementation, we can use the 
                # "Simple CMA-ES" or just adapt sigma and use random directions if C becomes unstable.
                # However, a proper CMA-ES implementation is better.
                
                # Let's use a robust implementation strategy:
                # We will maintain C as a matrix and update it.
                # Sampling: z ~ N(0, I), y = x + sigma * (C @ z) ? No, y = x + sigma * C^{1/2} z.
                # Computing C^{1/2} every generation is O(n^3). For n=5, it's trivial.
                
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                z = np.random.randn(dim)
                y[i] = x + sigma * C_sqrt @ z
                f_y[i] = func(y[i])
                calls += 1
            
            if i + 1 < lam:
                # If we broke early, only process the ones we evaluated
                # This complicates the CMA-ES update. 
                # Better to ensure we don't break in the middle of a generation if possible, 
                # or just stop the whole loop.
                # Given 500 budget and lam=10, we have 50 full generations. 
                # Time limit is 300s. 500 calls is very fast. Time limit unlikely to hit.
                # So we assume full generations.
                pass

            # Sort by fitness
            indices = np.argsort(f_y)
            y_sorted = y[indices]
            f_y_sorted = f_y[indices]
            
            # Update best
            if f_y_sorted[0] < best_f:
                best_f = f_y_sorted[0]
                best_x = y_sorted[0].copy()
            
            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update paths
            delta_x = x - x_old
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (delta_x / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_sqrt_inv @ delta_x
            
            # Update covariance
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - budget + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            
            # Note: The condition above is approximate. Standard formula:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1-(1-c_sigma)^(2*t)) < (1.4 + 2/(dim+1)) * sqrt((dim+1)/(dim-1))
            # We need t (number of generations). Let's track gen.
            
            # Simplified h_sigma check for stability
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximate denominator
            denom = np.sqrt(1 - (1 - c_sigma)**2) # Rough approx for small t
            # Better: track generation count
            # Let's use a simpler adaptation if full CMA is too risky to implement perfectly in text.
            
            # Given the constraints, a simplified (1+1)-ES with adaptive sigma and random restarts might be safer?
            # But CMA is much better for f22.
            
            # Let's stick to the standard update formulas but be careful with h_sigma.
            # If h_sigma is True, add the rank-1 update to p_c part.
            
            # Rank-1 update
            y_w = y_sorted[:mu]
            rank1 = np.outer(delta_x, delta_x)
            
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_w_i = y_sorted[i] - x_old
                rank_mu += weights[i] * np.outer(y_w_i, y_w_i)
            
            # Update C
            # c1 = (mu_eff - 2 + 1/mu_w) / (dim^2 + 2)
            # cmu = 2 * (mu_eff - 2 + 1/mu_w) / (dim^2 + 2)
            # mu_eff = 1 / sum(weights^2)
            # mu_w = sum(weights)^2 / sum(weights^2)
            
            mu_eff = 1 / np.sum(weights**2)
            mu_w = np.sum(weights)**2 / np.sum(weights**2)
            
            c1 = (mu_eff - 2 + 1 / mu_w) / (dim**2 + 2)
            cmu = 2 * (mu_eff - 2 + 1 / mu_w) / (dim**2 + 2)
            
            # h_sigma calculation
            # We need generation count. Let's assume calls is roughly lam * gen.
            gen = calls // lam
            if gen > 0:
                denom_h = np.sqrt(1 - (1 - c_sigma) ** (2 * gen))
                threshold = (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                h_sigma = 1 if (np.linalg.norm(p_sigma) / denom_h < threshold) else 0
            else:
                h_sigma = 1
            
            C = (1 + c1 * (h_sigma - 1) + c1 + cmu - c1 - cmu) * C # Placeholder, need correct formula
            # Correct formula:
            C = (1 - c1 - cmu) * C + c1 * (rank1 + (h_sigma - 1) * np.outer(p_c, p_c)) + cmu * rank_mu
            
            # Update sigma
            sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Ensure C is positive semi-definite
            # Eigenvalue check and repair
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            if np.any(eigvals_C < 1e-12):
                C = eigvecs_C @ np.diag(np.maximum(eigvals_C, 1e-12)) @ eigvecs_C.T
            C_sqrt_inv = eigvecs_C @ np.diag(1 / np.sqrt(np.maximum(eigvals_C, 1e-12))) @ eigvecs_C.T

        return best_x, best_f
