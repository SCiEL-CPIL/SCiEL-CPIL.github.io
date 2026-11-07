import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        # Number of selected points
        mu = lambda_ // 2
        
        # Weights
        # Standard CMA-ES weights: w_i = log(mu + 0.5) - log(i)
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # Standard CMA-ES learning rates
        # c_sigma
        c_sigma = (mueff + 2) / (n + mueff + 5)
        # c_c
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        # c_1
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        # c_mu
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial state
        mean = np.zeros(n)
        sigma = 2.0  # Initial step size, covers [-2, 2] roughly, good for [-5, 5]
        C = np.eye(n)
        B = np.eye(n)
        D = 1.0
        
        # State variables for step size adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Cache for B and D (Cholesky of C)
        # We will recompute B and D every generation for simplicity in this short run
        
        calls = 0
        best_x = mean.copy()
        best_f = float('inf')
        
        # Helper to update Cholesky decomposition
        def update_chol(C):
            # Use np.linalg.cholesky
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # If not positive definite, add small regularization
                C_reg = C + 1e-8 * np.eye(n)
                L = np.linalg.cholesky(C_reg)
                return L
        
        # Initial Cholesky
        L = np.eye(n)
        
        while calls < budget:
            if time.time() - start_time > 280:
                break
                
            # Generate population
            population = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = np.random.randn(n)
                # y = sigma * L @ z
                y = sigma * (L @ z)
                x = mean + y
                # Clip to bounds
                x = np.clip(x, -5, 5)
                population[i] = x
                
            # Evaluate population
            f_values = np.zeros(lambda_)
            for i in range(lambda_):
                f_values[i] = func(population[i])
                calls += 1
                if calls >= budget:
                    break
                
            if calls >= budget:
                break
            
            # Sort by function value
            idx = np.argsort(f_values)
            population = population[idx]
            f_values = f_values[idx]
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = population[0].copy()
                
            # Update mean
            # mu_best = sum of weights * population[i]
            new_mean = np.sum(weights[:, np.newaxis] * population[:mu], axis=0)
            
            # Update step size and covariance
            # 1. Update p_sigma
            # Need B and D from previous iteration's C
            # For simplicity and robustness in this short budget, we can update C first, then use new B,D?
            # Standard CMA-ES updates p_sigma using old B, D.
            
            # Compute B and D from current C (before update)
            # We have L from previous iteration. 
            # Actually, let's just recompute B and D from C at the start of the update phase.
            try:
                L_old = np.linalg.cholesky(C)
                B_old = np.linalg.inv(L_old.T)
                D_old = np.sqrt(np.diag(C))
            except:
                B_old = np.eye(n)
                D_old = np.ones(n)
                
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) * B_old @ (mean - new_mean) / sigma
            # Note: mean is old mean, new_mean is the weighted average
            # The vector is (old_mean - new_mean) / sigma
            diff = (mean - new_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B_old @ diff)
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mueff) * B_old @ (mean - new_mean) / sigma
            # Wait, the formula for p_c uses the same diff but different coefficient?
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mueff) * (B @ (m_{old} - m_{new}) / sigma)
            # Yes.
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (B_old @ diff)
            
            # Update sigma
            # sigma = sigma * exp( c_sigma / d_sigma * ( ||p_sigma|| / chi_n - 1 ) )
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_c) * D^2 * I) + c_mu * sum w_i y_i y_i^T
            # h_c = ( ||p_c|| / sqrt(1 - (1 - c_c)^{2 * gen}) ) < (1.4 + 2/(n+1)) * chi_n
            # We need to track generations for h_c.
            # Let's approximate h_c = 1.0 if not converged, else 0.
            # For 50 generations, it's safe to assume h_c=1 initially?
            # Actually, let's implement a simple h_c check.
            # We need to store ||p_c|| history or just use the standard heuristic.
            # For simplicity and robustness, we can set h_c = 1.0.
            # This is a valid variant (CMA-ES with h_c=1).
            h_c = 1.0
            
            # y_i are the steps: (population[i] - mean) / sigma
            # But we need the weighted sum of outer products of y_i
            # y_i = (x_i - mean_old) / sigma
            # Note: We must use the OLD mean for the step calculation in the covariance update?
            # Yes, y_i = (x_i - m_{old}) / sigma
            # And we only use the first mu points.
            
            C_update = np.zeros((n, n))
            for i in range(mu):
                y_i = (population[i] - mean) / sigma
                C_update += weights[i] * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * np.diag(D_old**2)) + c_mu * C_update
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            
            # Update mean
            mean = new_mean
            
            # Update Cholesky for next iteration
            try:
                L = np.linalg.cholesky(C)
            except:
                C += 1e-8 * np.eye(n)
                L = np.linalg.cholesky(C)
                
        return best_x, best_f
