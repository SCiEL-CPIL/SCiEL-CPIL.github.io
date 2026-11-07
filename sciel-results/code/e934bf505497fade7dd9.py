"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        # CMA-ES parameters
        dim = self.dim
        pop_size = 4 + int(3 * np.log(dim))  # lambda
        step_size = 2.0  # initial sigma
        
        # Adaptation rates
        cc = (4 + dim / pop_size) / (dim + 4 + 2.0 / pop_size)
        cs = (pop_size / 2 + 1) / (dim + pop_size / 2 + 3)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmua = min(1 - c1, 2 / ((dim + 2) ** 2 + pop_size))
        damps = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs

        # Initialize state
        rng = np.random.default_rng(0)
        x_mean = rng.uniform(-5, 5, dim)
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        f_mean = func(x_mean)
        best_x = x_mean.copy()
        best_f = f_mean
        calls = 1
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Sample population
            X = np.zeros((pop_size, dim))
            F = np.zeros(pop_size)
            
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                # x = x_mean + step_size * C_sqrt * z
                # Since we maintain C_sqrt_inv, we need C_sqrt. 
                # Actually, standard CMA-ES maintains C and uses Cholesky or eigendecomposition.
                # For simplicity and speed in dim=5, we can use eigendecomposition every step or maintain C_sqrt.
                # Let's use eigendecomposition for C to get C_sqrt.
                # However, updating C_sqrt is tricky. 
                # Alternative: Use the fact that we can sample via z @ C_sqrt.
                # Let's just compute eigendecomposition of C once per generation to get C_sqrt.
                # In dim=5, this is cheap.
                break # Placeholder to break inner loop logic if we decide to do it outside
            # Actually, let's restructure to compute C_sqrt once per generation
            
            # Compute C_sqrt via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            # Note: This C_sqrt is symmetric, but C is symmetric PD. 
            # Sampling: x = x_mean + sigma * C_sqrt @ z
            
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x_i = x_mean + step_size * (C_sqrt @ z)
                x_i = np.clip(x_i, -5, 5)
                X[i] = x_i
                F[i] = func(x_i)
                calls += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i.copy()
                    
            # Sort by fitness
            order = np.argsort(F)
            mu = pop_size // 2
            
            # Weights
            w = np.zeros(mu)
            log_w = np.log(np.arange(1, mu + 1))
            w[:] = log_w / np.sum(log_w)
            w *= mu
            nu = 2 * min(1, (1 - np.sum(w**2)) / np.sum(w**2))
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += w[i] * X[order[i]]
            x_new_mean /= np.sum(w)
            
            # Update path moments
            y_sigma = (x_new_mean - x_mean) / step_size
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_sigma
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2) / np.sqrt(1 - (1 - cs)**(2*(calls//pop_size))) < (1.4 + 2/(dim+1)) * chi_n) else 0
            
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * y_sigma
            
            # Update covariance
            C = (1 - c1 - cmua) * C + c1 * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = (X[order[i]] - x_mean) / step_size
                C += cmua * w[i] * np.outer(y_i, y_i)
                
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update step size
            sigma = step_size * np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
            step_size = sigma
            
            # Update mean
            x_mean = x_new_mean
            
            # Check for convergence
            if np.linalg.norm(x_new_mean - x_mean) < 1e-12:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
