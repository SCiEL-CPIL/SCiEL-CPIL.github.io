import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        # CMA-ES Parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of parents
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff / 2 + 1) / (n + 4 + mu_eff / 2)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        L = np.eye(n)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = float('inf')
        
        calls_made = 0
        remaining = budget
        gen = 0

        while remaining >= lam:
            # Check time
            if time.time() - start_time > max_time:
                break
            
            gen += 1
            
            # Generate population
            Z = np.random.randn(lam, n)
            # Transform using Cholesky factor L
            Y = Z @ L.T
            X = x_mean + sigma * Y
            
            # Evaluate fitness
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(X[i])
                calls_made += 1
                remaining -= 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                    
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            
            # Select mu best individuals
            X_parents = X_sorted[:mu]
            
            # Recombination
            x_new = np.dot(w, X_parents)
            
            # Update sigma (CSA)
            z = (x_new - x_mean) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            
            # Hessian correction term
            # t is number of generations
            # Avoid division by zero or negative inside sqrt
            exp_term = (1 - c_sigma) ** (2 * gen / lam)
            denom_sq = 1 - exp_term
            if denom_sq < 0:
                denom_sq = 0
            denom = np.sqrt(denom_sq) if denom_sq > 0 else 1e-10
            
            threshold = (1.4 + 2 / (n + 1)) * np.sqrt((1 - c_sigma) / (1 + c_sigma))
            h_sigma = (np.linalg.norm(p_sigma) / denom < threshold)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Update Covariance Matrix
            delta = c_1 * (1 + (1 - c_sigma) * h_sigma * (2 - c_sigma)) + c_mu
            
            # Rank-1 update
            rank1 = np.outer(x_new - x_mean, x_new - x_mean)
            
            # Rank-mu update
            Y_parents = (X_parents - x_mean) / sigma
            y_mean = np.dot(w, Y_parents)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                diff = Y_parents[i] - y_mean
                rank_mu += w[i] * np.outer(diff, diff)
            
            C_new = (1 - delta) * C + delta * (rank1 / sigma**2 + rank_mu)
            
            # Symmetrize
            C_new = (C_new + C_new.T) / 2
            
            # Update L (Cholesky)
            try:
                L = np.linalg.cholesky(C_new)
            except np.linalg.LinAlgError:
                C_new = (C_new + C_new.T) / 2 + 1e-10 * np.eye(n)
                L = np.linalg.cholesky(C_new)
            
            C = C_new
            x_mean = x_new
            sigma = sigma * np.exp(c_sigma / mu_eff * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Prevent sigma from exploding or vanishing too fast
            if sigma < 1e-15:
                break

        return best_x, best_f
