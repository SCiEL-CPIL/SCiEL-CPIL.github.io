"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        if lam < 8:
            lam = 8
            
        # Weights for recombination
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Learning rates
        cc = (4 + lam / dim) / ((dim + 4) * (2 + lam / dim))
        cs = (lam / 2 + 1) / (dim + lam / 2 + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * (weights[1] - 1) / ((dim + 2) ** 2 + lam))
        
        # Initialization
        # Start from center or a random point? Random is better for global search initially.
        # But CMA-ES is local. Let's start from a random point in the box.
        rng = np.random.default_rng(42)
        
        # To maximize early progress, we can do a few random evaluations first 
        # or start CMA-ES from the best of a small initial set.
        
        # Initial set of points to find a good starting basin
        init_points = 10
        xs = rng.uniform(-5, 5, (init_points, dim))
        fs = np.array([func(x) for x in xs])
        
        best_idx = np.argmin(fs)
        best_x = xs[best_idx].copy()
        best_f = fs[best_idx]
        calls_made = init_points
        
        # If we still have budget, run CMA-ES from this best point
        if calls_made < budget:
            m = best_x.copy()
            sigma = 2.0 # Initial step size
            C = np.eye(dim)
            B = np.eye(dim)
            D = 1.0
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Precompute sqrtm for B, D
            def update_BD():
                nonlocal B, D
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive definiteness
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
                return B, D
            
            B, D = update_BD()
            
            while calls_made < budget and time.time() < deadline:
                # Sample lambda points
                # y_i ~ N(0, I)
                Y = rng.standard_normal((lam, dim))
                # Transform to x
                # x_i = m + sigma * B @ (D * y_i)
                # Note: B @ (D * Y.T).T is equivalent to (B * D) @ Y.T ?
                # C = B D^2 B^T. sqrt(C) = B D B^T.
                # x = m + sigma * (B * D) @ Y.T
                X = m + sigma * (B * D) @ Y.T
                X = X.T # Shape (lam, dim)
                
                # Clip to bounds
                X = np.clip(X, -5, 5)
                
                # Evaluate
                Fs = np.array([func(x) for x in X])
                calls_made += lam
                
                if calls_made > budget:
                    # We might have exceeded budget slightly if lam > remaining
                    # But the loop condition checks before start. 
                    # If we exceeded, we just break.
                    break
                
                # Sort by fitness
                order = np.argsort(Fs)
                X_sorted = X[order]
                Fs_sorted = Fs[order]
                
                # Update best
                if Fs_sorted[0] < best_f:
                    best_f = Fs_sorted[0]
                    best_x = X_sorted[0].copy()
                
                # Recombination
                # Weights are for the best mu individuals
                w = weights
                m_old = m.copy()
                m = np.sum(X_sorted[:mu] * w[:, None], axis=0)
                
                # Step-size adaptation (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / D @ (B.T @ (m - m_old) / sigma)
                # Note: The term (m - m_old) / sigma is the step in y-space?
                # Actually, standard CMA-ES update:
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) / sigma * B^-1 (m - m_old)
                # B^-1 = D^-1 B^T
                # So ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (D^-1 B^T (m - m_old)) / sigma
                
                # Correction: The standard formula is:
                # p_sigma <- (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * B^{-1} (m - m_old)
                # B^{-1} = D^{-1} B^T
                # So:
                step_in_y = (m - m_old) / sigma
                # Transform to y-space
                step_y = (D ** -1) * (B.T @ step_in_y)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * step_y
                
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - init_points) / lam)) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
                
                sigma *= np.exp(cs / lambda_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Covariance matrix adaptation
                pc = (1 - cc) * pc + hsig * cc / sigma * (m - m_old)
                
                # Rank 1 update
                C = (1 + c1 * hsig * (2 - hsig) - c1) * C + c1 * np.outer(ps, ps)
                
                # Rank mu update
                # Sum of outer products of weighted steps
                # y_i = B^{-1} (x_i - m_old) / sigma
                # We have X_sorted[:mu]
                X_diff = (X_sorted[:mu] - m_old) / sigma
                Y_diff = (D ** -1) * (B.T @ X_diff.T).T # Shape (mu, dim)
                
                C += cmu * np.sum(weights[:, None, None] * Y_diff[:, None, :] * Y_diff[:, :, None], axis=0)
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                # Occasionally re-orthonormalize B, D
                if calls_made % (5 * lam) == 0:
                    B, D = update_BD()
                    
                # Stopping criteria
                if sigma < 1e-10 or np.all(np.abs(X_sorted[0] - X_sorted[1]) < 1e-10):
                    break
                    
        return best_x, best_f

# EVOLVE-BLOCK-END
