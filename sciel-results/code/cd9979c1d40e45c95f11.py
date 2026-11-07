"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using multi-start Powell's method
with stratified Latin Hypercube initialization and L-BFGS-B polish.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Minimal CMA-ES Implementation
        # Population size
        lam = max(4 + int(3 * np.log(dim)), 10)
        # Initial standard deviation
        sigma = 1.0
        # Initial center
        x_mean = np.zeros(dim)
        
        # CMA-ES specific parameters
        c_sigma = (lam / 2 + 1) / (dim + 4 + lam / 2)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam / 2)
        mu = lam // 2  # Top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim ** 2))
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Main Loop
        while calls < budget and time.time() < deadline:
            # Generate population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                # Sample from multivariate normal with covariance C
                # Using Cholesky decomposition for efficiency
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    C = np.eye(dim)
                
                z = np.random.randn(dim)
                Y[i] = x_mean + sigma * (L @ z)
                # Clip to bounds
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                F[i] = func(Y[i])
                calls += 1
                update_best(Y[i], F[i])
            
            if calls >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update step-size control (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (L @ (x_new - x_mean) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) < (1 + 2 / (dim + 1)) * chi_n
            
            # Update covariance matrix (Rank-1 update for simplicity and robustness)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            C = (1 - c_1) * C + c_1 * (np.outer((x_new - x_mean) / sigma, (x_new - x_mean) / sigma))
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(dim)
            
            # Update mean and sigma
            x_mean = x_new
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clip sigma to prevent explosion
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
