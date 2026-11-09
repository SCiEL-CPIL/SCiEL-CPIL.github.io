import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters as per directive
        lambda_pop = 4
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d**2))
        
        best_f = float('inf')
        best_x = np.zeros(d)
        calls = 0
        rng = np.random.default_rng(42)
        
        # Restart logic
        max_restarts = 10
        restart_count = 0
        start_x = np.zeros(d)
        start_sigma = 5.0
        
        while calls < budget and time.time() < deadline and restart_count < max_restarts:
            # Initialize CMA-ES state
            x_m = start_x.copy()
            C = np.eye(d)
            C_12 = np.eye(d)
            sigma = start_sigma
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            
            gen = 0
            # Run CMA-ES until stagnation or budget/time
            while calls < budget and time.time() < deadline:
                # Sample
                Z = rng.standard_normal((lambda_pop, d))
                Y = np.dot(Z, C_12.T)
                X = x_m + sigma * Y
                
                # Evaluate
                F = np.array([func(x) for x in X])
                calls += lambda_pop
                
                # Update best
                min_idx = np.argmin(F)
                if F[min_idx] < best_f:
                    best_f = F[min_idx]
                    best_x = X[min_idx].copy()
                    
                # Sort
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                Z = Z[order]
                
                # Update mean
                y_w = np.dot(weights, X[:mu].T).T
                
                # Update sigma path
                y_sigma = np.dot(C_12.T, y_w / sigma)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * y_sigma
                
                # h_sigma condition
                gen_count = gen + 1
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * gen_count))) / chi_n
                h_sigma = 1.0 if h_sigma < 1.4 * (1 - c_sigma) else 0.0
                
                # Update covariance path
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c)) * y_w
                
                # Update covariance matrix
                v = np.array(X[:mu] - x_m).T
                H = np.zeros((d, d))
                for i in range(mu):
                    H += weights[i] * np.outer(v[:, i], v[:, i])
                
                H = H / sigma**2
                
                C = (1 - c_1 - c_mu) * C \
                    + c_1 * ((1 + (1 - h_sigma) * c_c * (2 - c_c)) * np.outer(p_c, p_c) - C) \
                    + c_mu * H
                    
                # Regularize and Cholesky
                C = (C + C.T) / 2.0
                C += 1e-12 * np.eye(d)
                try:
                    C_12 = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    break
                    
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update mean
                x_m = x_m + sigma * y_w
                
                gen += 1
                
                # Stagnation/Convergence check
                if sigma < 1e-8 or sigma > 1000.0:
                    break
                cond_C = np.linalg.cond(C)
                if cond_C > 1e12:
                    break
                
            # If we broke out, prepare for restart
            restart_count += 1
            if calls < budget and time.time() < deadline:
                # Restart from best found so far with a larger sigma
                start_x = best_x.copy()
                start_sigma = 2.0 # Moderate restart sigma
                
        return best_x, best_f
