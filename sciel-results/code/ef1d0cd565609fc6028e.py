"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Time guard: 300s limit, leave 10s margin for safety
        start_time = time.time()
        max_time = 290.0
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Initial sigma based on box size
        sigma = 2.0 
        x = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        
        # Strategy parameters
        mu = max(1, n // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))
        
        best_x = x.copy()
        best_f = float('inf')
        
        def get_evals(x, sigma):
            """Generate mu+lambda candidates and evaluate them."""
            lam = max(4, 4 + int(3 * n))
            # Generate lambda random directions
            N = np.random.standard_normal((lam, n))
            # Sample
            Y = x[:, None] + sigma * (C_sqrt @ N.T).T
            # Clip to box
            Y = np.clip(Y, -5.0, 5.0)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            
            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Take top mu
            Y = Y[:mu]
            F = F[:mu]
            
            # Mean
            mean = np.sum(weights[:, None] * Y, axis=0)
            
            return mean, F[0], Y, F, lam

        # Initial evaluation
        f_start = func(x)
        if f_start < best_f:
            best_f = f_start
            best_x = x.copy()
            
        calls_made = 1
        
        # Main loop
        while calls_made < budget:
            if time.time() - start_time > max_time:
                break
            
            # Generate and evaluate
            mean, f_best_child, Y, F, lam = get_evals(x, sigma)
            calls_made += lam
            
            if f_best_child < best_f:
                best_f = f_best_child
                best_x = Y[0].copy()
            
            # Update step size
            diff = mean - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * C_sqrt.T @ (diff / sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff / sigma
            
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - n - 1)))) / chi_n < 1.4 + 2 / (n + 1)
            h_sigma = float(h_sigma)
            
            delta_h = (1 - h_sigma) * np.sqrt((1 - c_c) ** 2 * (2 - c_c) * mueff)
            
            C = (1 + c_1 * h_sigma * delta_h**2 - c_1 - c_mu) * C \
                + c_1 * np.outer(p_c, p_c) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(Y - x, Y - x) / sigma**2, axis=0)
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2
            
            # Update C_sqrt via SVD
            # Note: For small n, this is fast. For large n, it might be slower, but n=5 here.
            U, s, Vt = np.linalg.svd(C)
            C_sqrt = U @ np.diag(np.sqrt(s)) @ Vt
            
            x = mean
            
        return best_x, best_f
# EVOLVE-BLOCK-END
