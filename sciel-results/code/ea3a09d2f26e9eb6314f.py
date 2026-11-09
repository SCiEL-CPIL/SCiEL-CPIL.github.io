"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin before 300s limit
        dim = self.dim
        rng = np.random.default_rng(42)

        # Initialize CMA-ES parameters
        # Start at origin as a reasonable guess for the center, though shift is hidden.
        # CMA-ES is robust to initialization if given enough budget, but origin is a safe default.
        mean = np.zeros(dim)
        
        # Initial step size. 
        # Bent cigar is very flat in one direction and steep in another.
        # A moderate initial sigma is best.
        sigma = 2.0
        
        # Population size
        # Standard CMA-ES uses lambda = 4 + floor(3 log n)
        lam = 4 + int(3 * np.log(dim))
        
        # Weights for recombination
        # Standard weights: log(2:lambda/2) - log(1:lambda) normalized
        mu = lam // 2  # Number of weights
        w = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(lam, 0, -1))
        w /= np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Strategy parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Evolution path
        p_c = np.zeros(dim)
        p_sigma = np.zeros(dim)
        
        # Cholesky factorization update frequency
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial best
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Precompute some constants
        # We will use a simplified CMA-ES update to ensure speed and stability.
        # Full CMA-ES with dense matrix updates is O(n^3). For n=5, this is trivial.
        
        def update_covariance():
            nonlocal C, C_inv, C_sqrt, C_inv_sqrt
            # Update C
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1 - delta_c) * C) + cmu * sum(wi * yi @ yi.T)
            # Note: The standard update is slightly different. 
            # Let's use the standard rank-mu update.
            
            # We need the evolution steps y_i
            # This is handled in the main loop.
            pass # Placeholder, logic below
            
        # Main loop
        while calls < self.budget and time.time() < deadline:
            # Generate population
            ys = np.zeros((lam, dim))
            xs = np.zeros((lam, dim))
            fs = np.zeros(lam)
            
            for i in range(lam):
                # Sample
                z = rng.standard_normal(dim)
                # Transform
                y = C_sqrt @ z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                
                ys[i] = y
                xs[i] = x
                fs[i] = func(x)
                calls += 1
                
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = x.copy()
                    
                # Check deadline inside inner loop to be safe
                if time.time() > deadline - 10: 
                    return best_x, best_f

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update mean
            mean_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
            
            # Update sigma (CSA)
            # y_sigma = (1/sigma) * (mean_new - mean)
            y_sigma = (mean_new - mean) / sigma
            
            # Update evolution paths
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (C_inv_sqrt @ y_sigma)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ y_sigma)
            
            # Update covariance matrix
            # Rank-one update
            delta_cs = 1 + 2 * max(0, (np.linalg.norm(p_sigma) / chi_n - 1) / (dim + 2)) - 1
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**((calls + 1) / (cs * (dim + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # Rank-mu update
            delta_c = (1 - h_sigma) * cc * (2 - cc)
            
            # Compute the rank-mu term
            # sum(wi * (yi - (1-h_sigma)*y_sigma) @ (yi - (1-h_sigma)*y_sigma).T)
            # Actually, the standard formula is:
            # C = (1 - c1 - cmu + delta_c) * C + c1 * (p_c @ p_c.T + (1 - delta_c) * C) + cmu * sum(wi * yi @ yi.T)
            # Wait, the term is usually:
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1 - delta_c) * C) + cmu * Y @ W @ Y.T
            # Where Y is the matrix of selected steps.
            
            # Let's stick to the simpler, robust update often used in implementations:
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1 - delta_c) * C) + cmu * sum(wi * ys_i @ ys_i.T)
            
            # Note: ys_i here should be the standardized steps y_i = (x_i - mean_old) / sigma
            # We stored ys in the loop.
            
            # Recalculate ys for selected individuals (they are already in ys_sorted)
            # Note: ys_sorted contains the y vectors corresponding to the sorted xs.
            
            # Rank-mu term
            # We only use the top mu
            Y_sel = ys_sorted[:mu]
            W = w[:mu]
            
            # Sum of outer products weighted by w
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu_term += W[i] * np.outer(Y_sel[i], Y_sel[i])
                
            # Rank-one term
            # p_c is already updated.
            # The term is p_c @ p_c.T + (1 - delta_c) * C
            # But the standard formula is:
            # C_new = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1 - delta_c) * C) + cmu * RankMuTerm
            # This simplifies to:
            # C_new = (1 - c1 - cmu + c1 * (1 - delta_c)) * C + c1 * (p_c @ p_c.T) + cmu * RankMuTerm
            # Let's just compute it directly to avoid algebra errors.
            
            C_old = C
            C = (1 - c1 - cmu) * C_old + c1 * (np.outer(p_c, p_c) + (1 - delta_c) * C_old) + cmu * rank_mu_term
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            mean = mean_new
            
            # Update sigma
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C_inv, C_sqrt, C_inv_sqrt periodically or every step?
            # For dim=5, every step is fine.
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_inv = np.linalg.inv(C)
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Regularization
                C += 1e-4 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt = L
                    C_inv = np.linalg.inv(C)
                    C_inv_sqrt = np.linalg.inv(L)
                except:
                    # Fallback to identity if it breaks
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                    sigma = 2.0
                    mean = best_x
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
