"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with budget and time constraints."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initial population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial mean: random point in the box
        rng = np.random.default_rng(42)
        mean = rng.uniform(-5, 5, dim)
        
        # Initial step size
        # Standard initialization for CMA-ES
        sigma = 2.0 
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution path for covariance adaptation
        pc = np.zeros(dim)
        # Evolution path for step size adaptation
        ps = np.zeros(dim)
        
        # Parameters
        mu = lambda_ // 2  # Number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Learning rates
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Chi-squared approximation for sigma adaptation
        chi_lambda = np.sqrt(lambda_ * (1 - 1/(4*lambda_) - 1/(21*lambda_**2)))
        
        # Initial evaluation
        f_mean = func(mean)
        best_x = mean.copy()
        best_f = f_mean
        calls = 1
        
        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = np.zeros((lambda_, dim))
            f_offspring = np.zeros(lambda_)
            
            # Compute eigenvalues for whitening
            # For small dim, eigendecomposition is fast
            try:
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.maximum(eigval, 1e-12)
                D = 1.0 / np.sqrt(eigval)
                B = eigvec
                # Transform: B * diag(sqrt(D))
                # Actually, standard CMA-ES uses B * diag(1/sqrt(eigval))
                # Let's use the standard update rule
            except:
                D = np.ones(dim)
                B = np.eye(dim)
                
            # Generate samples
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Apply transformation
                y = B @ (D * z)
                x = mean + sigma * y
                # Clip to box
                x = np.clip(x, -5, 5)
                offspring[i] = x
                f_offspring[i] = func(x)
                calls += 1
                
                if f_offspring[i] < best_f:
                    best_f = f_offspring[i]
                    best_x = x.copy()
                    
            # Sort by fitness
            idx = np.argsort(f_offspring)
            offspring = offspring[idx]
            f_offspring = f_offspring[idx]
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * offspring[:mu], axis=0)
            
            # Update evolution paths
            # Inverse square root of C
            try:
                # C^{-1/2} = B * diag(1/sqrt(eigval)) * B.T
                C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(eigval)) @ B.T
            except:
                C_inv_sqrt = np.eye(dim)
                
            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
            
            # pc update
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lambda_)))) < (1 + 2 / (dim + 1)) * chi_lambda
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
            
            # Covariance update
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = C_inv_sqrt @ (offspring[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)
                
            C = (1 + c_1 * h_sigma * (2 - c_c) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Regularization (ensure positive definite)
            # Simple fix: add small identity if needed, or re-eigen if not PD
            # For dim=5, this is cheap
            try:
                eigval, eigvec = np.linalg.eigh(C)
                if np.any(eigval < 1e-12):
                    C = C + 1e-12 * np.eye(dim)
            except:
                C = C + 1e-12 * np.eye(dim)
                
            mean = new_mean
            f_mean = f_offspring[0] # Not strictly needed, but good for tracking
            
            # Sigma adaptation
            sigma = sigma * np.exp(c_sigma / chi_lambda * (np.linalg.norm(ps) / chi_lambda - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 100.0)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
