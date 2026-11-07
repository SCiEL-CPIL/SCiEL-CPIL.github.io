"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: multi-start CMA-ES with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Multi-start strategy: try a few different initial points
        # But we only have 300 evaluations total, so we need to be careful.
        # Let's use a single robust CMA-ES-like approach with good initialization.
        
        # Initialize near center with some randomness
        x = rng.uniform(-1, 1, dim)
        fx = func(x)
        
        # CMA-ES parameters
        n = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = n // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        sigma = 2.0  # initial step size
        C = np.eye(dim)  # covariance matrix
        pc = np.zeros(dim)  # evolution path for covariance
        ps = np.zeros(dim)  # evolution path for sigma
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        calls_used = 1
        best_x = x.copy()
        best_fx = fx
        
        # Main loop
        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            B = rng.standard_normal((n, dim))
            # Transform using C^{-1/2}
            # Use Cholesky decomposition for C
            try:
                L = np.linalg.cholesky(C)
                Linv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                Linv = np.eye(dim)
            
            # Generate candidates
            ys = []
            fys = []
            for i in range(n):
                y = x + sigma * (Linv @ B[i])
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fys.append(fy)
                calls_used += 1
                
                # Check time during population generation
                if time.time() > deadline:
                    break
            
            if calls_used >= budget or len(ys) < mu:
                break
            
            # Sort by fitness
            order = np.argsort(fys)
            ys = [ys[i] for i in order]
            fys = [fys[i] for i in order]
            
            # Update best
            if fys[0] < best_fx:
                best_fx = fys[0]
                best_x = ys[0].copy()
            
            # Mean
            m = sum(weights[i] * ys[i] for i in range(mu))
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Linv @ (m - x))
            
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)
            
            # Update pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (Linv @ (m - x))
            
            # Update C
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used // n)))) < (1.4 + 2 / (dim + 1)) * chi_n
            delta_hsig = 1.0 if hsig else 0.0
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) - delta_hsig * C) \
                + cmu * sum(weights[i] * np.outer(ys[i] - x, ys[i] - x) for i in range(mu))
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
            
            x = m
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
