"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Custom CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Popsize lambda
        lambda_ = 4 + int(3 * np.log(dim))
        # Number of weights
        mu = lambda_ // 2
        
        # Weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = -np.log(i + 1)
        w /= w.sum()
        
        # Mu_eff
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial values
        x0 = np.zeros(dim)
        sigma = 1.0  # Initial step size
        
        # State variables
        mean = x0.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Eigen decomposition cache
        D_inv = np.ones(dim)
        B = np.eye(dim)
        
        # Evaluation counter
        eval_count = 0
        best_x = x0.copy()
        best_f = float('inf')
        
        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return 1e10
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val
        
        # Initial evaluation
        best_f = eval_func(x0)
        
        # Main Loop
        while eval_count < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            # Update eigen decomposition if needed (every generation for simplicity in this small dim)
            B, D = np.linalg.eigh(C)
            D_inv = 1.0 / D
            
            Y = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size=lambda_)
            X = mean + sigma * (Y @ (B * D_inv).T)
            
            # Evaluate population
            F = np.array([eval_func(x) for x in X])
            
            if eval_count >= budget:
                break
                
            # 2. Sort by fitness
            indices = np.argsort(F)
            X_sorted = X[indices]
            F_sorted = F[indices]
            
            # 3. Update mean
            new_mean = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # 4. Update path accumulators
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff / dim)
            c_c_sqrt = np.sqrt(c_c * (2 - c_c) * (dim + mu_eff / 2) / dim)
            
            # Standard deviation of the mean of the selected points
            sigma_mean = np.sqrt(np.sum(w**2) * np.trace(C))
            
            ps = (1 - c_sigma) * ps + c_sigma_sqrt * (B * D_inv) @ ((new_mean - mean) / (sigma * np.sqrt(np.sum(w**2))))
            pc = (1 - c_c) * pc + c_c_sqrt * (B * D_inv) @ ((new_mean - mean) / sigma)
            
            # 5. Update covariance matrix
            # Rank-one update
            C += c_1 * (np.outer(pc, pc) + (1 if np.dot(ps, ps) > (1 + 4 / dim) * (2 - c_c) * dim else 0) * np.eye(dim) - C)
            
            # Rank-mu update
            delta_sigma = np.dot(ps, ps) / dim - (1 - c_sigma)
            C += c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer((X_sorted[:mu] - mean) / sigma, (X_sorted[:mu] - mean) / sigma), axis=0)
            
            # 6. Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.sqrt(np.dot(ps, ps)) / np.sqrt(dim) - 1))
            
            # 7. Update mean
            mean = new_mean
            
            # Ensure C is positive definite (symmetrize)
            C = (C + C.T) / 2
            
            # Check for convergence or stagnation
            # If the variance is extremely small, we might be stuck
            # But with 500 budget, we just run until budget is exhausted or time runs out
            
        return best_x, best_f
# EVOLVE-BLOCK-END
