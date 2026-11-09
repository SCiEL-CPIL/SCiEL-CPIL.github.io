"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Standard recommendations:
        # lambda = 4 + floor(3 * ln(n))
        # For n=5: lambda = 4 + floor(3*1.609) = 4 + 4 = 8
        # mu = lambda // 2 = 4
        # Weights: w_i = ln(mu+0.5) - ln(i) for i=1..mu, normalized
        # c_sigma, c_c, c_1, c_mu based on standard formulas
        
        lambda_pop = 8
        mu = lambda_pop // 2
        
        # Calculate weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 2)) + c_sigma
        
        # Covariance adaptation parameters
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial sigma
        # Box is [-5, 5]^5. Width is 10.
        # A good initial sigma is often 0.3 * range or similar.
        # Let's start with sigma = 1.0 which is reasonable for a 10-unit box.
        sigma = 1.0
        
        # Initial mean: center of the box
        x_mean = np.zeros(d)
        
        # Initial covariance: Identity
        C = np.eye(d)
        
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Cholesky factor of C
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(d)
        
        D = np.prod(np.sqrt(np.diag(C)))
        
        # History for ranking
        history_x = []
        history_f = []
        
        calls = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Initial evaluation of the mean
        f_mean = func(x_mean)
        calls += 1
        best_f = f_mean
        best_x = x_mean.copy()
        
        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate lambda offspring
            y_list = []
            x_list = []
            f_list = []
            
            for i in range(lambda_pop):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                z = np.random.standard_normal(d)
                y = B @ z
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                calls += 1
                
                y_list.append(y)
                x_list.append(x)
                f_list.append(fx)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if not f_list:
                break
            
            # Sort by fitness
            idx = np.argsort(f_list)
            y_sorted = np.array([y_list[i] for i in idx[:mu]])
            x_sorted = np.array([x_list[i] for i in idx[:mu]])
            f_sorted = np.array([f_list[i] for i in idx[:mu]])
            
            # Update mean
            x_mean = np.dot(weights, x_sorted)
            
            # Update evolution path for sigma
            y_w = np.dot(weights, y_sorted)
            h_sigma = 1.0 if np.sqrt(np.sum(y_w**2) / (1 - (1 - np.exp(-1.0 / d_sigma)))**2) < (1.4 + 2.0 / (d + 1)) * np.sqrt((d + 1) / (d - 1)) else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c)
            
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) - C - delta_h) \
                + c_mu * np.sum([w * np.outer(y, y) for w, y in zip(weights, y_sorted)], axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Regularize C to ensure positive definiteness
            # Add small identity if needed
            try:
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += (1e-12 - np.min(eigvals)) * np.eye(d)
            except:
                C += 1e-12 * np.eye(d)
            
            # Update Cholesky factor
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity and B to identity
                C = np.eye(d)
                B = np.eye(d)
            
            D = np.prod(np.sqrt(np.diag(C)))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1.0))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-12, 5.0)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
