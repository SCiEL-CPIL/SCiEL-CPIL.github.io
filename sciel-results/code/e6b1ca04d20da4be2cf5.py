"""Optimized program for BBOB f12 (Bent Cigar) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.
    CMA-ES is highly effective for ill-conditioned functions like Bent Cigar
    because it adapts the covariance matrix to align with the valley structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Time safety guard
        start_time = time.time()
        deadline = start_time + 320.0  # 320s out of 330s limit

        # CMA-ES Parameters
        # Initial mean: Center of the box
        x = np.zeros(d)
        
        # Initial sigma: Large enough to explore the box [-5, 5]^5
        # Box width is 10, so sigma ~ 2.5 covers most of the space
        sigma = 2.5
        
        # Population size: Default is 4 + floor(3 log d)
        # For d=5, this is 4 + floor(3*1.6) = 4 + 4 = 8
        # Larger population helps with exploration but costs more evals.
        # Budget is 500. 500 / 8 = 62.5 generations. This is reasonable.
        # Let's try a slightly larger population for better valley alignment early on.
        lam = 10 
        
        # Weights for recombination
        mu = lam // 2  # Top half of the population
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Damping for covariance update
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        
        # State variables
        pc = np.zeros(d)  # Evolution path for sigma
        ps = np.zeros(d)  # Evolution path for covariance
        C = np.eye(d)     # Covariance matrix
        B = np.eye(d)     # Eigenvectors
        D = np.ones(d)    # Eigenvalues
        inv_D = np.ones(d)
        inv_sqrt_C = np.eye(d)
        
        # Precompute constants for efficiency
        chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        
        # Loop until budget or time is exhausted
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop_x = []
            pop_f = []
            
            for i in range(lam):
                # Standard normal
                z = np.random.standard_normal(d)
                # Transform using Cholesky-like decomposition (B * D * z)
                # Note: In standard CMA-ES, we use B * D * z where C = B D^2 B^T
                y = B @ (D * z)
                xi = x + sigma * y
                # Clip to box [-5, 5]
                xi = np.clip(xi, -5, 5)
                
                fi = func(xi)
                calls += 1
                pop_x.append(xi)
                pop_f.append(fi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                    
                # Early exit if time is tight
                if time.time() > deadline:
                    break
            
            # If we ran out of time mid-population, we can't update properly
            # We just return the best found so far.
            if len(pop_x) < lam:
                break
                
            # Sort population by fitness
            order = np.argsort(pop_f)
            pop_x_sorted = np.array(pop_x)[order]
            pop_f_sorted = np.array(pop_f)[order]
            
            # Update mean
            x_old = x.copy()
            x = np.sum([weights[i] * pop_x_sorted[i] for i in range(mu)], axis=0)
            
            # Update evolution paths
            # pc update
            pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (x - x_old)
            
            # ps update
            ps = (1 - c_c) * ps + np.sqrt(c_c * (2 - c_c) * mueff) * inv_sqrt_C @ ((x - x_old) / sigma)
            
            # Sigma update (1/5 success rule variant)
            sigma *= np.exp(c_sigma / chiN * (np.linalg.norm(ps) / chiN - 1))
            
            # Covariance matrix update
            # Rank-one update
            v = (x - x_old) / sigma
            C = (1 + c_1 * mueff * (1 - c_sigma) - c_1) * C + c_1 * np.outer(v, v)
            
            # Rank-mu update
            z_sorted = np.array([(pop_x_sorted[i] - x_old) / sigma for i in range(mu)])
            z_weighted = z_sorted.T @ weights
            # We need to transform z by inv_sqrt_C
            z_weighted_inv = (inv_sqrt_C @ z_weighted.T).T
            C += c_mu * (np.outer(z_weighted_inv, z_weighted_inv) - 
                         (1 - c_sigma) * (np.eye(d) * np.sum(z_weighted_inv**2, axis=1)).T)
            
            # Ensure positive definiteness and update decomposition
            # Simple re-decomposition every 10 generations or if ill-conditioned
            # For simplicity and robustness in this short budget, we re-decompose every generation
            # This is O(d^3) which is tiny for d=5.
            try:
                D2, B = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                D2 = np.maximum(D2, 1e-12)
                D = np.sqrt(D2)
                inv_D = 1.0 / D
                inv_sqrt_C = (B @ np.diag(inv_D)) @ B.T
            except:
                # If decomposition fails, reset covariance to identity
                C = np.eye(d)
                B = np.eye(d)
                D = np.ones(d)
                inv_D = np.ones(d)
                inv_sqrt_C = np.eye(d)

        return best_x, best_f
# EVOLVE-BLOCK-END
