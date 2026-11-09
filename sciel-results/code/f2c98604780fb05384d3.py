"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Global-Local Optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def _evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Sampling to find the best basin
        # We use a mix of structured points and random points.
        # The Discus function has a global minimum that is shifted.
        # We need to explore the box [-5, 5]^d efficiently.
        
        # 1. Origin and axis-aligned points
        init_points = [np.zeros(dim)]
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                init_points.append(pt)
        
        # 2. Random points in the box
        # Sample 20 random points
        for _ in range(20):
            pt = rng.uniform(-5.0, 5.0, size=dim)
            init_points.append(pt)

        # Evaluate initial points
        # Limit initial phase to 20% of budget to leave room for CMA-ES
        max_init = min(len(init_points), max(10, budget // 5))
        for pt in init_points[:max_init]:
            if call_count >= budget - 50: # Reserve some budget for CMA-ES
                break
            if time.time() > deadline:
                break
            _evaluate(pt)
        
        # If we haven't found anything, use origin (shouldn't happen)
        if best_x is None:
            best_x = np.zeros(dim)
            _evaluate(best_x)

        # Phase 2: CMA-ES for local exploitation
        # Start CMA-ES from the best point found
        
        x_mean = best_x.copy()
        
        # CMA-ES Parameters
        # For dim=5, a population size of 10-15 is often good for speed
        lambda_ = 4 * dim + 4 # 24 for dim=5
        mu_ = lambda_ // 2
        
        # Weights
        w = np.zeros(mu_)
        for i in range(mu_):
            w[i] = np.log(mu_ + 0.5) - np.log(i + 1)
        w /= w.sum()
        
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
        
        # Initial sigma: 
        # Since we found a good point, we can start with a smaller sigma
        # to exploit the local neighborhood. 
        # A sigma of 0.5-1.0 is usually good if the point is close to optimum.
        # If the point is far, CMA-ES will adapt.
        sigma = 1.0
        
        # Covariance matrix C
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Eigen decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Chi_n constant for sigma update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Main CMA-ES Loop
        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(lambda_):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                f_y = _evaluate(y)
                offspring.append((y, f_y, z))
            
            if len(offspring) < mu_:
                break
                
            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu_]
            
            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * y for i, (y, f, z) in enumerate(selected)], axis=0)
            
            # Update evolution paths
            # Approximate inverse square root of C for efficiency in low dim
            # Or just use B and D
            C_inv_sqrt = B @ np.diag(1.0 / np.maximum(D, 1e-10)) @ B.T
            delta_mean = x_mean - old_mean
            
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ delta_mean)
            
            # Update sigma
            z_mean = np.sum([w[i] * z for i, (y, f, z) in enumerate(selected)], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z_mean
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update Covariance Matrix
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_sigma) ** 2) * np.eye(dim))
            
            for i in range(mu_):
                diff = selected[i][0] - x_mean
                C += c_mu * w[i] * np.outer(diff, diff)
            
            C = (C + C.T) / 2  # Symmetrize
            
            # Ensure positive semi-definite
            try:
                eigvals_C, _ = np.linalg.eigh(C)
                if np.min(eigvals_C) < 1e-12:
                    C = C + (1e-10 - np.min(eigvals_C)) * np.eye(dim)
            except:
                C = np.eye(dim)
                
            # Update eigen decomposition for next iteration
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(eigvals, 1e-12))
                B = eigvecs
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
