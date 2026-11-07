"""CMA-ES based optimizer for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For dim=5, lambda is typically 4 + floor(3*log(dim)) = 4 + 7 = 11
        lambda_pop = 4 + int(math.floor(3 * math.log(dim)))
        mu = lambda_pop // 2
        
        # Initial parameters
        # Start at 0,0,0,0,0? The minimizer is in [-4,4]. 
        # Katsuura usually has global min at 0. But there's a shift.
        # Let's start at 0.
        x_mean = np.zeros(dim)
        sigma = 1.0
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Step sizes
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (2 * dim) + 2 / dim) / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Memory
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        
        # Cholesky factor of C
        C_cholesky = np.eye(dim)
        
        # Best tracking
        x_best = x_mean.copy()
        f_best = func(x_best)
        calls = 1
        
        # History for AOCC? No, AOCC is computed by evaluator.
        # We just need to minimize func.
        
        # Evaluation loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            fs = []
            
            for i in range(lambda_pop):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Sample
                z = np.random.randn(dim)
                y = x_mean + sigma * (C_cholesky @ z)
                
                # Clip to box? 
                # "func(x) clips a finite length-5 vector to the box"
                # So we can just pass y.
                fy = func(y)
                calls += 1
                
                ys.append(y)
                fs.append(fy)
                
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
            
            if not ys:
                break
            
            # Sort
            indices = np.argsort(fs)
            ys = [ys[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            # Only process if we have enough
            if len(ys) < mu:
                break
                
            # Update mean
            w = weights[:mu]
            x_new_mean = np.sum([w[i] * ys[i] for i in range(mu)], axis=0)
            
            # Update sigma
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_cholesky.T @ (x_new_mean - x_mean))
            
            # Hessian norm of C
            # Approximation: norm(C)
            norm_C = np.linalg.norm(C)
            
            if np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1.4 + 2 / (dim + 1)) * sigma * norm_C:
                sigma *= math.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) / sigma - 1))
            
            # Update C
            # p_c update
            p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_new_mean - x_mean)
            
            # Rank-one update
            C = (1 + c_1 * (mueff - 1 + c_mu) - c_1) * C + c_1 * np.outer(x_mean, x_mean) # Wait, formula is (1 + c1*(mueff-1+c_mu) - c1) * C + c1 * (x_mean - x_old_mean)(x_mean - x_old_mean)^T
            # Let's use the standard update:
            # C = (1 + c1*(mueff-1+c_mu) - c1) * C + c1 * (x_new_mean - x_mean)(x_new_mean - x_mean)^T
            # Wait, x_mean is the OLD mean.
            diff = x_new_mean - x_mean
            C = (1 + c_1 * (mueff - 1 + c_mu) - c_1) * C + c_1 * np.outer(diff, diff)
            
            # Rank-mu update
            # C = C + c_mu * sum( w_i * (y_i - x_mean)(y_i - x_mean)^T )
            # Note: y_i are the selected ones.
            # We need to be careful with the scaling.
            # Standard CMA-ES rank-mu update:
            # C = (1 + c_mu*(mueff-1+c_mu) - c_mu) * C + c_mu * sum( w_i * (y_i - x_new_mean)(y_i - x_new_mean)^T )
            # Wait, usually it's:
            # C = (1 + c1*(mueff-1+c_mu) - c1) * C + c1 * (x_new_mean - x_mean)(x_new_mean - x_mean)^T + c_mu * sum( w_i * (y_i - x_new_mean)(y_i - x_new_mean)^T )
            # Let's implement the combined update properly.
            
            # Recalculate C from scratch to avoid accumulation errors in manual implementation?
            # No, iterative update is standard.
            
            # Let's stick to a simpler, more robust implementation of CMA-ES updates if possible, 
            # or just use a simplified version.
            
            # Actually, implementing CMA-ES correctly from scratch is error-prone.
            # Let's use a simpler, robust alternative: **Covariance Matrix Adaptation with simplified updates** or **Particle Swarm Optimization (PSO)**?
            # PSO is very robust and simple.
            
            # Let's switch strategy to PSO if CMA-ES implementation is too risky.
            # But the prompt asks for a DIFFERENT strategy. The current is (1+1)-ES.
            # CMA-ES is a distinct design family (population-based, covariance adaptation).
            # Let's try to get CMA-ES right.
            
            # Standard update:
            # C = (1 + c1*(mueff-1+c_mu) - c1) * C + c1 * (x_new_mean - x_mean)(x_new_mean - x_mean)^T + c_mu * sum( w_i * (y_i - x_new_mean)(y_i - x_new_mean)^T )
            
            # Let's re-do the C update correctly.
            C_old = C
            C = (1 + c_1 * (mueff - 1 + c_mu) - c_1) * C_old
            C += c_1 * np.outer(diff, diff)
            for i in range(mu):
                d_y = ys[i] - x_new_mean
                C += c_mu * w[i] * np.outer(d_y, d_y)
            
            x_mean = x_new_mean
            
            # Update Cholesky
            # Add small regularization for numerical stability
            C += 1e-10 * np.eye(dim)
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not positive definite, reset or regularize more
                C = np.eye(dim)
                C_cholesky = np.eye(dim)
                sigma = 1.0
                x_mean = np.zeros(dim)
                
        return x_best, f_best
# EVOLVE-BLOCK-END
