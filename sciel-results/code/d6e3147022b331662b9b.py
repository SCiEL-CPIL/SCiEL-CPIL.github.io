import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0  # Safety margin for 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 10 
        # Initial sigma
        sigma = 1.0 
        # Initial mean
        x_mean = np.zeros(dim)
        
        # Covariance matrix C
        C = np.eye(dim)
        # Cholesky factor of C
        C_chol = np.eye(dim)
        
        # Step size control parameters
        mu = lam // 2  
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        # Damping for step size
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        x_mean = np.zeros(dim)
        f_mean = func(x_mean)
        best_x = x_mean.copy()
        best_f = f_mean
        calls = 1
        
        if calls >= budget:
            return best_x, best_f

        gen = 0
        
        while calls < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
                
            gen += 1
            
            # Generate lambda candidates
            X = np.zeros((lam, dim))
            F = np.zeros(lam)
            
            for i in range(lam):
                z = np.random.randn(dim)
                y = C_chol @ z
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                X[i] = x
                F[i] = func(x)
                calls += 1
                
                if calls >= budget:
                    break
                    
            if calls >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                
            # Update mean
            x_new_mean = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update covariance matrix and step size
            
            # Recover y_i for the top mu points
            y_sorted = (X_sorted[:mu] - x_mean) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * np.sum(weights[:, None] * y_sorted, axis=0)
            
            # Calculate h_sigma
            # Threshold
            threshold = (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(mu_eff + 1.0)
            # Denominator: sqrt(1 - (1 - c_sigma)^(2 * gen))
            denom = np.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * gen))
            if denom > 1e-12:
                norm_p_sigma = np.linalg.norm(p_sigma) / denom
            else:
                norm_p_sigma = 0.0
                
            h_sigma = 1.0 if norm_p_sigma < threshold else 0.0
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + h_sigma * c_c * sqrt(mu_eff) * (x_new_mean - x_mean) / sigma
            # Note: (x_new_mean - x_mean) / sigma is the weighted average of y_sorted
            y_mean = np.sum(weights[:, None] * y_sorted, axis=0)
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * y_mean
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim^2))
            e_norm = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (32.0 * dim ** 2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / e_norm - 1.0))
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - (1 - c_sigma)^2 * h_sigma) * sigma^2 * C) 
            #    + c_mu * sum(weights[i] * y_i * y_i.T)
            
            # Sum of outer products of y_sorted
            S = np.zeros((dim, dim))
            for i in range(mu):
                S += weights[i] * np.outer(y_sorted[i], y_sorted[i])
                
            # Update C
            C = (1.0 - c_1 - c_mu) * C + \
                c_1 * (np.outer(p_c, p_c) + (1.0 - (1.0 - c_sigma) ** 2 * h_sigma) * C) + \
                c_mu * S
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update Cholesky factor
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, C is not positive definite.
                # Add a small diagonal component
                C = C + 1e-6 * np.eye(dim)
                C_chol = np.linalg.cholesky(C)
                
            # Update mean
            x_mean = x_new_mean

        return best_x, best_f
