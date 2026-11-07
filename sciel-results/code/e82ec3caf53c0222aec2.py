"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB-style f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        
        # Initial point - use center for Schwefel-like functions
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Step size
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Learning rates
        c_sigma = (1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1)) / (n + 4)
        c_c = (4 + lam / n) / (n + 4 + 2 * lam / n)
        c_1 = 2 / ((n + 1.3) ** 2 + lam / 2)
        c_mu = min(1 - c_1, 2 * ((2 + np.log(n / 2)) ** 2 - 1) / ((n + 2) ** 2 + lam / 2))
        
        # Weights for recombination
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1)) - np.mean(np.log(np.arange(1, mu + 1)))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w ** 2)
        
        # Chi-n
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Call counter
        calls_made = 1
        
        for gen in range(budget - 1):
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Generate population
            offspring = []
            offspring_vals = []
            
            # Generate random directions
            for i in range(lam):
                # Generate from N(0, C)
                z = np.linalg.cholesky(C).T @ np.random.randn(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            for y in offspring:
                if time.time() > deadline - 5:
                    break
                fy = func(y)
                calls_made += 1
                offspring_vals.append(fy)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            
            if len(offspring_vals) < lam:
                break
            
            # Sort offspring
            idx = np.argsort(offspring_vals)
            sorted_offspring = [offspring[i] for i in idx]
            sorted_vals = [offspring_vals[i] for i in idx]
            
            # Update mean
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += w[i] * sorted_offspring[i]
            
            # Evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * C_inv @ (y_w - x)
            norm_ps = np.linalg.norm(ps) / chi_n
            
            hsig = 1.0 if norm_ps < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + 1))) else 0.0
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (y_w - x)
            
            # Update covariance
            delta_d = hsig * (1 / c_c - 1) + 1
            rank_mu = np.outer(y_w, y_w)
            rank_1 = np.outer(pc, pc)
            
            C = (1 + c_1 * (delta_d - 1) - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            
            # Ensure symmetry and positive definiteness
            C = 0.5 * (C + C.T)
            
            # Update sigma
            sigma *= np.exp(c_sigma / 5 * (norm_ps / chi_n - 1))
            
            # Update mean
            x = y_w
            fx = sorted_vals[0]
            
            # Update C_inv periodically (every 10 generations)
            if gen % 10 == 0 or gen == 0:
                try:
                    C_inv = np.linalg.inv(C)
                except:
                    C_inv = np.eye(n)
                    C = np.eye(n)
            
            # Check if we've used our budget
            if calls_made >= budget:
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
