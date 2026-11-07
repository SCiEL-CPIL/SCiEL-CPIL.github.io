import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        rng = np.random.default_rng(0)
        
        # CMA-ES Parameters
        # For dim=5, lambda=10 or 15 is good. 500/15 = 33 generations.
        # Let's use lambda=10 for slightly larger populations per generation or more generations.
        # Actually, for Rosenbrock, larger lambda helps exploration. Let's try lambda=15.
        # 500 / 15 = 33 generations.
        # If we use lambda=10, 50 generations.
        # Standard CMA-ES lambda is 4 + floor(3 ln n). For n=5, 4 + 3*1.6 = 8.8 -> 9.
        # Let's stick to a standard-ish lambda, maybe 12 or 15.
        # Let's use lambda=15 to ensure good coverage.
        
        n = dim
        lambda_ = 15
        mu = lambda_ // 2
        
        # Initial Mean
        m = np.zeros(n)
        # Initial Sigma
        sigma = 2.0
        
        # Covariance Matrix
        C = np.eye(n)
        # Inverse Cholesky for sampling
        # We need to update Cholesky factorization.
        # For n=5, we can just compute Cholesky every step or use a robust update.
        # Simpler: Compute Cholesky of C every generation. n=5 is tiny.
        
        # Path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Weights
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + 4)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Damping
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        x_best = m.copy()
        fx_best = func(m)
        calls = 1
        
        # Track best found overall
        best_x = m.copy()
        best_f = fx_best
        
        gen = 0
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sample lambda_ points
            # Cholesky decomposition of C
            # Ensure C is symmetric positive definite
            C_sym = (C + C.T) / 2.0
            try:
                B = np.linalg.cholesky(C_sym)
            except np.linalg.LinAlgError:
                # Add small jitter if singular
                C_sym += 1e-8 * np.eye(n)
                B = np.linalg.cholesky(C_sym)
            
            # Standard Normal samples
            Z = rng.standard_normal((lambda_, n))
            # Transform to X
            X = m[None, :] + sigma * (Z @ B.T)
            
            # Evaluate
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                x_i = np.clip(X[i], -5, 5)
                F[i] = func(x_i)
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i.copy()
            
            # If we ran out of budget in the middle of evaluation, break
            if calls >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update Mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update Path p_sigma
            # Note: We use the inverse square root of C for the path update
            # C_inv_sqrt = np.linalg.inv(B) @ B.T # This is not quite right for inverse sqrt
            # Better: C_inv_sqrt = np.linalg.inv(B) is not C^(-1/2)
            # C = B @ B.T. C^(-1/2) is not simply B^(-1).
            # However, for the path update in standard CMA-ES, we use C^(-1/2) (m - m_old) / sigma.
            # Since n=5, we can compute C^(-1/2) via eigen decomposition or just use the Cholesky inverse if we approximate.
            # Actually, the standard update is:
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * C_inv_sqrt * (m - m_old) / sigma
            # Let's compute C_inv_sqrt properly.
            # Eigen decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C_sym)
            # Clip negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (m - m_old) / sigma)
            
            # Check if p_sigma is too large (sigma update)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(calls//lambda_ + 1)))) < (1.4 + 2/(n+1)) * chi_n
            
            # Update Sigma
            if h_sigma:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            else:
                # If h_sigma is false, we might still want to update sigma but standard CMA-ES uses the condition
                # Actually, standard CMA-ES updates sigma every generation.
                # The h_sigma is for the correction term in c_c.
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update Path p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (m - m_old) / sigma)
            
            # Update Covariance Matrix C
            # Rank-1 update
            if h_sigma:
                rank1 = np.outer(p_c, p_c)
            else:
                rank1 = np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C # Wait, the correction is added to the rank-1 term?
                # Formula: (1 - (1-c_c)^2) * C is added to rank1? No.
                # The term is: c_1 * (rank1 + (1 - (1-c_c)^2) * C * (1-h_sigma) ... )
                # Let's stick to the standard formula:
                # C = (1 - c_1 - c_mu + mu_eff * c_mu * (1 - h_sigma) * (1 - c_c)^2) * C + ...
                # This is getting complex to get right without reference.
                # Let's use a simpler, robust rank-1 + rank-mu update.
                
            # Let's re-implement the C update carefully.
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * (1 - c_c)^2 * C) + c_mu * sum w_i (x_i - m_old)(x_i - m_old)^T
            
            # Calculate Rank-Mu update
            # Use X_sorted[:mu] - m_old
            D = X_sorted[:mu] - m_old[None, :]
            # Weighted sum of outer products
            rank_mu = np.sum(weights[:, None, None] * D[:, :, None] * D[:, None, :], axis=0)
            
            # Combine
            # Term 1: Decay
            decay = 1 - c_1 - c_mu
            # Term 2: Rank 1
            # If h_sigma, use p_c p_c^T. If not, use p_c p_c^T + (1 - (1-c_c)^2) C ?
            # The standard CMA-ES update for C is:
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma)(1 - c_c)^2 C) + c_mu * RankMu
            # Wait, the (1 - h_sigma)(1 - c_c)^2 C term is inside the c_1 part?
            # Yes, to ensure positive definiteness when h_sigma=0.
            
            if h_sigma:
                rank1_term = np.outer(p_c, p_c)
            else:
                rank1_term = np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C_sym
            
            C = decay * C_sym + c_1 * rank1_term + c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
            
            gen += 1
            
            # If we are stuck (no improvement for many generations), we might restart or perturb, 
            # but with only 500 calls, we should just let it run.
            
        return best_x, best_f
