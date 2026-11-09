"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        # CMA-ES parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial mean
        x = np.zeros(dim)
        
        # Initial step size
        # Heuristic: 1/3 of the box width is a reasonable start for [-5, 5]
        sigma = 10.0 / 3.0 
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Parameters
        c_sigma = (lam/2 + 1) / (dim + 2 + lam/2)
        mu = lam // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1.2) * (dim + 1.2) + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / (dim + 1)) / ((dim + 2) * (dim + 2) + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        best_x = x
        best_f = np.inf
        
        # Initial evaluation
        best_f = func(x)
        best_x = x.copy()
        
        remaining_budget = self.budget - 1
        
        while remaining_budget > 0:
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            # Update C if necessary (simplified: we update C every generation)
            # For efficiency in Python, we might skip some updates or use simplified CMA-ES
            # Here we implement a standard CMA-ES step
            
            # Sample population
            # We need to generate y_i ~ N(0, C)
            # Efficiently: y = L @ z, where C = L @ L.T
            L = np.linalg.cholesky(C)
            
            pop = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = L @ z
                x_i = x + sigma * y
                # Clip to box [-5, 5]
                x_i = np.clip(x_i, -5, 5)
                pop.append((x_i, func(x_i)))
                remaining_budget -= 1
                if remaining_budget <= 0:
                    break
            
            if remaining_budget <= 0:
                break
                
            # Sort by fitness
            pop.sort(key=lambda t: t[1])
            
            # Update best
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()
                
            # Update mean
            x_new = np.zeros(dim)
            for i in range(min(mu, len(pop))):
                x_new += weights[i] * pop[i][0]
            x = x_new
            
            # Calculate y_old for path updates
            y_old = (x - x_new) / sigma if sigma > 0 else np.zeros(dim)
            
            # Update p_sigma
            h_sigma = (np.sqrt((1 - (1 - c_sigma)**2 * remaining_budget)) / (1 - c_sigma**remaining_budget)) * np.sqrt(1 - c_sigma)
            # Simplified h_sigma check
            h_sigma = 1.0 # Constant for simplicity in this constrained implementation
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x - x_new)
            
            # Update p_c
            h_c = (np.sqrt((1 - (1 - c_c)**2 * remaining_budget)) / (1 - c_c**remaining_budget)) * np.sqrt(1 - c_c)
            # Simplified h_c check
            h_c = 1.0
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_new)
            
            # Update C
            delta_h = (1 - h_sigma) * c_sigma * np.linalg.norm(p_sigma) + h_sigma - 1
            C = (1 + c_1 + c_mu * delta_h - c_1 - c_mu) * C + c_1 * (p_c @ p_c - C)
            
            # Rank-mu update
            for i in range(min(mu, len(pop))):
                y_i = (pop[i][0] - x_new) / sigma if sigma > 0 else np.zeros(dim)
                C += c_mu * weights[i] * np.outer(y_i, y_i)
                
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # If C becomes singular or non-positive definite, reset it
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                sigma = 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
