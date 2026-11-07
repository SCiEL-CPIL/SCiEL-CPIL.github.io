import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        d = self.dim
        rng = np.random.default_rng(42)
        
        # Initial point: center of the box
        m = np.zeros(d)
        f_m = func(m)
        best_x = m.copy()
        best_f = f_m
        
        # CMA-ES parameters
        sigma = 2.5  # Larger initial step for bent cigar
        C = np.eye(d)
        
        # Precompute inverse square root and Cholesky
        eigvals, eigvecs = np.linalg.eigh(C)
        Cinv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
        Cholesky = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # Evolution paths
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Standard CMA-ES parameters for mu=1
        mueff = 1.0
        cc = (2 + 2 / np.sqrt(d + 1.3)) / (d + 4 + 2 / np.sqrt(d + 1.3))
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.7)**2 + mueff)
        dsig = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        calls = 1
        stagnation_count = 0
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            z = rng.standard_normal(d)
            x = m + sigma * (Cholesky @ z)
            x = np.clip(x, -5, 5)
            
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Update paths
            y = x - m
            
            # ps update (in z-space)
            z_step = Cinv_sqrt @ (y / sigma)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_step
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1.0 if norm_ps / chiN < 1.4 + 2 / (d + 1) else 0.0
            sigma *= np.exp(cs / dsig * (norm_ps / chiN - 1))
            
            if fx < f_m:
                # Success
                f_m = fx
                m = x
                stagnation_count = 0
                
                # pc update (in scaled x-space, i.e., y/sigma)
                pc = (1 - cc) * pc + cc * (y / sigma)
                
                # C update
                C = (1 - c1) * C + c1 * (np.outer(pc, pc) + (1 - h_sigma) * cc * (2 - cc) * C)
                
            else:
                # Failure
                stagnation_count += 1
                pc = (1 - cc) * pc
                
                # C update (no rank-1 term)
                C = (1 - c1) * C + c1 * ((1 - h_sigma) * cc * (2 - cc) * C)
                
                # Stagnation-based sigma increase
                if stagnation_count >= 30:
                    sigma *= 2.0
                    stagnation_count = 0
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 15.0)
            
            # Renormalize C for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            Cinv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            Cholesky = eigvecs @ np.diag(np.sqrt(eigvals))
        
        return best_x, best_f
