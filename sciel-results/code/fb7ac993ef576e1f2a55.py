"""CMA-ES based optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 and cumulative step-size control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size lambda = 4
        lam = 4
        # Number of selected individuals mu = 2 (half the population)
        mu = lam // 2
        # Weights for the mu best individuals
        # Standard CMA-ES weights: w_i = log(i + 0.5) - log(mu + 0.5) for i=1..mu
        # Then normalize so sum w_i = 1
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(i + 1) - np.log(mu + 0.5)
        # Actually the standard formula is w_i = ln(i + 0.5) - ln(mu/2 + 0.5) for i=0..mu-1
        # Let's use the simpler standard initialization
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(i + 1) - np.log(mu + 0.5)
        # Ensure positive weights and normalize
        weights = np.maximum(weights, 0)
        if np.sum(weights) == 0:
            weights = np.ones(mu)
        weights = weights / np.sum(weights)
        
        # mu_eff = 1 / sum(w_i^2)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial values
        # Start from origin as per directive
        x_mean = np.zeros(n)
        sigma = 1.0
        
        # Covariance matrix C = I
        C = np.eye(n)
        # Inverse square root of C for sampling
        # Initially C = I, so C^(-1/2) = I
        # We need to maintain C and its inverse square root
        
        # Cumulation path for step size
        p_sigma = np.zeros(n)
        # Cumulation path for covariance
        p_c = np.zeros(n)
        
        # Precompute some values
        # For efficient sampling, we can use Cholesky decomposition or eigenvalue decomposition
        # Since n=5, we can afford eigenvalue decomposition each step if needed, 
        # but let's use a simpler approach: maintain C and use np.linalg.cholesky or similar
        
        # Actually, for n=5, we can just sample using the covariance matrix directly
        # by transforming standard normal samples
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        
        call_count = 1
        
        # Precompute weights for selection
        # We need to sort the population and pick the mu best
        
        # Main loop
        while call_count < budget:
            # Sample lambda individuals
            # We need to sample from N(x_mean, sigma^2 * C)
            # Sample z ~ N(0, I), then x = x_mean + sigma * C^(1/2) @ z
            
            # For simplicity and efficiency with small n, we can use
            # the fact that if C = B @ B.T (Cholesky), then C^(1/2) = B
            # But C might not be positive definite due to numerical issues
            # Let's use eigenvalue decomposition for robustness
            
            # Actually, let's maintain C and use a helper to sample
            # For n=5, eigenvalue decomposition is fast enough
            
            # Compute C^(1/2) via eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues to 0
            eigvals = np.maximum(eigvals, 0)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Sample lambda points
            pop = np.zeros((lam, n))
            for j in range(lam):
                z = np.random.standard_normal(n)
                pop[j] = x_mean + sigma * C_sqrt @ z
            
            # Evaluate population
            fvals = np.zeros(lam)
            for j in range(lam):
                x_j = np.clip(pop[j], -5, 5)
                fvals[j] = func(x_j)
                call_count += 1
                
                # Track best
                if fvals[j] < best_f:
                    best_f = fvals[j]
                    best_x = x_j.copy()
                
                # If we've used all budget, break
                if call_count >= budget:
                    break
            
            if call_count >= budget:
                break
            
            # Sort population by fitness
            sorted_idx = np.argsort(fvals)
            # Take the mu best
            idx = sorted_idx[:mu]
            
            # Compute new mean
            # x_mean_new = sum(w_i * x_{i:lam})
            x_mean_new = np.zeros(n)
            for i in range(mu):
                x_mean_new += weights[i] * pop[idx[i]]
            
            # Update step size
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C^(-1/2) @ (x_mean_new - x_mean)
            # C^(-1/2) = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            eigvals_safe = np.maximum(eigvals, 1e-20)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_safe)) @ eigvecs.T
            
            diff = x_mean_new - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ diff)
            
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(it+1)}) < (1.4 + 2/(n+1)) * sqrt((mu_eff+1)/(mu_eff-1))
            # For simplicity, use a simplified version
            # ||p_sigma||
            ps_norm = np.linalg.norm(p_sigma)
            # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            # Threshold
            h_sigma = 1 if ps_norm / chi_n < (1.4 + 2/(n+1)) else 0
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
            
            # Rank-one update: (1 - c_1 - c_mu) * C
            # Rank-mu update: c_mu * sum(w_i * (y_i * y_i.T)) where y_i = C^(-1/2) * (x_{i:lam} - x_mean) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1 - h_sigma * (2 - c_c)) * C)
            
            # Rank-mu update
            for i in range(mu):
                y_i = C_inv_sqrt @ (pop[idx[i]] - x_mean) / sigma
                C += c_mu * weights[i] * (y_i @ y_i.T)
            
            x_mean = x_mean_new
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Check for numerical stability
            # If eigenvalues become too small or too large, reset
            eigvals_check = np.linalg.eigvalsh(C)
            if np.any(eigvals_check < 1e-10) or np.any(eigvals_check > 1e10):
                # Reset covariance matrix
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = 1.0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
