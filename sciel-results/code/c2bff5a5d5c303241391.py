"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For dim=5, lambda is typically around 4 + floor(3 * log(dim))
        # 4 + 3*1.6 = 8.8 -> 8
        lam = max(4 + int(3 * np.log(dim)), 2)
        
        # Initial step size
        # BBOB functions often have a specific scale. Starting at 2.0 is reasonable for [-5,5].
        sigma = 2.0
        
        # Mean (start at center or random? Center is often good for symmetric functions, 
        # but BBOB f6 is shifted. Random start within box is safer to avoid bias if shift is large,
        # but 0,0,0,0,0 is the center. Let's start at 0 to maximize initial coverage radius relative to center.
        # Actually, the problem says minimizer is in [-4,4]^5. Starting at 0 is a good prior.
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse square root of C (for sampling)
        # Use Cholesky decomposition or eigen decomposition for stability
        # For dim=5, eigen is fine and robust
        def invsqrt(mat):
            eigvals, eigvecs = np.linalg.eigh(mat)
            # Clip small negative eigenvalues due to numerical errors
            eigvals = np.maximum(eigvals, 1e-12)
            inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
            return eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T
        
        C_inv_sqrt = invsqrt(C)
        
        # Evolution path for sigma
        cs = (lam + 2) / (dim + lam + 5)
        ps = np.zeros(dim)
        
        # Evolution path for covariance
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + lam)
        cmu = min(1 - c1, 2 * (2 / (dim + 1.3) - 1) / ((dim + 2) ** 2 + lam))
        
        # Success history rule parameters
        hsig = 1 - 2 / (lam + 2) + 2 / ((dim + lam) ** 2 - 4)
        hsig = max(0, min(1, hsig))
        
        # Initial evaluation
        best_f = np.inf
        best_x = m.copy()
        
        # Evaluate initial mean
        m_clipped = np.clip(m, -5, 5)
        f_m = func(m_clipped)
        if f_m < best_f:
            best_f = f_m
            best_x = m_clipped.copy()
            
        calls_made = 1
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline - 5.0:
                break
                
            # Sample lambda candidates
            # z ~ N(0, I)
            Z = np.random.randn(lam, dim)
            # y = C_inv_sqrt @ z
            Y = Z @ C_inv_sqrt.T  # Since C is symmetric, C_inv_sqrt is symmetric
            # x = m + sigma * y
            X = m[None, :] + sigma * Y
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate candidates
            F = np.array([func(x_i) for x_i in X])
            calls_made += lam
            
            # Sort by fitness
            I = np.argsort(F)
            X_sorted = X[I]
            F_sorted = F[I]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                
            # Update mean
            # Weights for mean update
            # Use log weights or equal weights? Log weights are standard for CMA-ES.
            # mu = lam // 2
            mu = lam // 2
            # Weights w_i = log(i+1) - log(mu) for i < mu
            # Normalized
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(i + 1) - np.log(mu)
            w = w / np.sum(w)
            
            # Effective population size
            mueff = 1.0 / np.sum(w ** 2)
            
            # New mean
            m_old = m.copy()
            m = m + np.sum([w[i] * (X_sorted[i] - m) for i in range(mu)], axis=0)
            
            # Update sigma (HCR)
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) / sigma * C_inv_sqrt @ (m - m_old)
            # Note: C_inv_sqrt is applied to the step (m - m_old)
            step = (m - m_old)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ step)
            
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lam + 1))) < (1.4 + 2 / (dim + 1)) * chiN:
                sigma *= np.exp(cs / lam * (np.linalg.norm(ps) / chiN - 1))
                
            # Update covariance
            # C = (1 - c1 - cmu) * C + c1 * (ps @ ps.T + (1 - (1 - cs)**2) * C_old) + cmu * sum(w_i * y_i @ y_i.T)
            # We need C_old for the ps term.
            C_old = C.copy()
            
            # Term 1
            C = (1 - c1 - cmu) * C
            
            # Term 2 (ps contribution)
            # The formula is slightly different depending on the implementation.
            # Standard: c1 * (ps @ ps.T + (1 - (1 - cs)**2) * C_old)
            C += c1 * (np.outer(ps, ps) + (1 - (1 - cs) ** 2) * C_old)
            
            # Term 3 (rank-mu update)
            # y_i = (x_i - m_old) / sigma
            Y_rank = (X_sorted[:mu] - m_old[None, :]) / sigma
            # C += cmu * sum(w_i * y_i @ y_i.T)
            # Efficiently: W_sqrt = sqrt(w)
            w_sqrt = np.sqrt(w)
            Y_weighted = Y_rank * w_sqrt[:, None]
            C += cmu * (Y_weighted.T @ Y_weighted)
            
            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2
            
            # Numerical stability: project to PSD
            # For dim=5, eigen decomposition is cheap
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2
            
            # Update inverse square root
            C_inv_sqrt = invsqrt(C)
            
            # Check for stagnation or convergence
            # If sigma is very small or variance is low, break?
            # The budget is the main constraint.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
