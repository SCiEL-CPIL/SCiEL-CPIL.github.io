"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for robust
optimization under unknown affine transformations.
"""
import time
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        rng = np.random.default_rng(12345)

        dim = self.dim
        budget = self.budget

        def still_ok():
            return time.time() < deadline

        # Phase 1: Initialization
        # Sample 20 points to find a good starting point
        n_init = 20
        best_x = np.zeros(dim)
        best_f = np.inf
        xs = []
        for _ in range(n_init):
            x = rng.uniform(-5.0, 5.0, dim)
            f = float(func(x))
            xs.append((x, f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Use the best point as initial mean
        m = best_x.copy()
        
        # Estimate initial step size
        # If we found a point close to 0, shrink sigma. Otherwise keep it large.
        # Standard CMA-ES sigma0 is often 1.0 or 2.0.
        # Given box [-5,5], 2.0 is reasonable.
        sigma = 2.0
        
        # CMA-ES parameters
        # lambda = population size
        lam = 4 + int(3 * np.log(dim))
        # mu = number of parents
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        C_inv_sqrt = np.eye(dim)
        
        # Step size control
        # Standard CMA-ES parameters
        # c_sigma, d_sigma
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        
        # c_c, d_c
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        d_c = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_c
        
        # c_1, c_mu
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x_overall = best_x.copy()
        best_f_overall = best_f
        
        calls_made = n_init
        
        while calls_made < budget and still_ok():
            # Generate lambda samples
            ys = []
            xs_pop = []
            fs_pop = []
            
            for i in range(lam):
                # Sample y from N(0, C)
                # y = C_inv_sqrt @ z
                z = rng.standard_normal(dim)
                y = C_inv_sqrt @ z
                x_new = m + sigma * y
                # Clip to box
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = float(func(x_new))
                calls_made += 1
                
                ys.append(y)
                xs_pop.append(x_new)
                fs_pop.append(f_new)
                
                if f_new < best_f_overall:
                    best_f_overall = f_new
                    best_x_overall = x_new.copy()
                    
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(fs_pop)
            ys = [ys[i] for i in idx]
            xs_pop = [xs_pop[i] for i in idx]
            fs_pop = [fs_pop[i] for i in idx]
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys[i]
            
            # Update step size
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2(iter+1)}) < (1 + 2/dim) * chi_n
            # Approximation: just check norm
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ y_w
            
            # Update p_c
            # h_c condition
            # ||p_c|| / sqrt(1 - (1 - c_c)^{2(iter+1)}) < (1 + 2/dim) * chi_n
            # We don't track iter exactly, so use a simpler heuristic or fixed threshold
            # For simplicity and robustness, we use a standard threshold
            h_c = 1.0 # Simplified
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mueff) * y_w
            
            # Update C
            # Delta h_c
            delta_h_c = (1 - h_c) * c_c * (2 - c_c)
            
            # Rank-1 update
            rank1 = np.outer(p_c, p_c)
            # Rank-mu update
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = ys[i]
                rankmu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * delta_h_c - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Eigenvalue check/update
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues
            eigvals = np.clip(eigvals, 1e-12, 1e12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update C_inv_sqrt
            # C_inv_sqrt = (C_inv_sqrt @ C) ^ (-1/2)
            # More efficiently: C_inv_sqrt = (C @ C_inv_sqrt.T) ^ (-1/2) ?
            # Standard update: C_inv_sqrt = (C_inv_sqrt @ C) ** (-1/2) is not quite right.
            # We want C_inv_sqrt such that C_inv_sqrt @ C_inv_sqrt.T = C^{-1}
            # Update rule: C_inv_sqrt = (C_inv_sqrt @ C) ** (-1/2)
            # Let's compute C_inv_sqrt from scratch every few steps or use the update
            # C_inv_sqrt = (C_inv_sqrt @ C) ** (-1/2)
            # Actually, the standard update is:
            # C_inv_sqrt = (C_inv_sqrt @ C) ** (-1/2)
            # But C changes.
            # Let's just recompute C_inv_sqrt using eigendecomposition of C
            # Since dim=5, this is cheap.
            eigvals, eigvecs = np.linalg.eigh(C)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update mean
            m = m + sigma * y_w
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Prevent sigma from exploding or vanishing
            sigma = np.clip(sigma, 1e-12, 10.0)

        return best_x_overall, float(best_f_overall)
