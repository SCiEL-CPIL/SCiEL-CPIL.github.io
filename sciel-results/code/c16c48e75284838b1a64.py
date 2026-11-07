"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # 290s margin within 300s limit (evaluator kills at 330s, task says 300s)
        # Using 290s to be safe against overhead.
        deadline = start_time + 290 

        best_x = None
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        # Larger population improves covariance estimation in early generations
        # for ill-conditioned problems, as observed in successful variants.
        lam = 16 
        mu = lam // 2  # number of parents
        # Weights for the selected parents
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size adaptation parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Covariance matrix adaptation parameters
        c1 = (2 - 1 / mu_eff + 2 * mu_eff / dim) / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial mean and step size
        # Start at center, with a step size that covers the likely basin [-4, 4]^5
        mean = np.zeros(dim)
        sigma = 2.2 
        
        # Initial covariance matrix (identity)
        C = np.eye(dim)
        # Inverse square root of C for transformation
        # Initially C=I, so C_inv_sqrt=I
        C_inv_sqrt = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute Cholesky decomposition for sampling
        # We update C and its decomposition every generation
        
        # Helper to update Cholesky decomposition
        def update_cholesky(C_mat):
            try:
                L = np.linalg.cholesky(C_mat)
                return L
            except np.linalg.LinAlgError:
                # Fallback to eigen decomposition if not PD
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-12, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                return L

        L = update_cholesky(C)
        
        # Main loop
        while total_calls < budget and time.time() < deadline:
            # Generate population
            # Sample N(0, I)
            zs = np.random.randn(lam, dim)
            # Transform to N(mean, sigma^2 C)
            # y = sigma * L @ z + mean
            # Note: L is lower triangular from Cholesky
            # We need to compute L @ z for each row
            # Efficient: (L @ zs.T).T
            ys = (L @ zs.T).T * sigma + mean
            
            # Evaluate population
            fs = np.array([safe_call(y) for y in ys])
            
            if total_calls >= budget:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(ys[:mu] * w[:, np.newaxis], axis=0)
            
            # Update evolution path for sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ (mean - old_mean) / sigma)
            
            # h_sigma: check if p_sigma is small enough
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_lam = np.sqrt(lam) * (1 - 1 / (2 * lam**2) + 1 / (8 * lam**4))
            h_sigma = 1 if norm_p_sigma / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * chi_lam else 0
            
            # Update evolution path for C
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma
            
            # Rank-one and rank-mu update of C
            # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (1 - h_sigma) * cc * (2 - cc) * C) + cmu * sum(w_i * (y_i - old_mean)(y_i - old_mean)^T)
            
            # Precompute delta
            delta = mean - old_mean
            
            # Update C
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cc * (2 - cc) * C)
            
            # Rank-mu update
            # We need the standardized differences
            # (ys[:mu] - old_mean) / sigma
            diff = (ys[:mu] - old_mean) / sigma
            # Weighted outer product sum
            # sum(w_i * diff_i diff_i^T)
            # Efficient: (diff.T @ (w[:, np.newaxis] * diff)).T ? No.
            # Let D = diff (mu x dim)
            # W = w (mu,)
            # Term = D.T @ (W[:, np.newaxis] * D)
            term = diff.T @ (w[:, np.newaxis] * diff)
            C += cmu * term
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2.0
            
            # Update Cholesky decomposition
            L = update_cholesky(C)
            
            # Update C_inv_sqrt
            # We need the inverse square root of C.
            # C = L L^T. C^{-1/2} = (L^{-1/2})^T L^{-1/2} ?
            # Actually, if C = Q D Q^T, then C^{-1/2} = Q D^{-1/2} Q^T.
            # Using Cholesky: C = L L^T.
            # We can compute C_inv_sqrt via eigen decomposition of C for stability, 
            # or use the fact that if we maintain C, we can just recompute.
            # Given dim=5, eigen decomposition is cheap and robust.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Adapt sigma
            sigma *= np.exp(cs / damps * (norm_p_sigma / chi_lam - 1))
            
            # Ensure sigma doesn't blow up or vanish
            if sigma > 100:
                sigma = 100
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
