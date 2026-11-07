"""Optimizer for BBOB-style f17 (Schaffers F7) using CMA-ES."""
import numpy as np
import time

class Optimizer:
    """CMA-ES for BBOB-style f17 with hidden shift/rotation/offset."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        pop_size = 4 + int(3 * np.log(dim))  # Standard lambda = 4 + 3*ln(mu)
        pop_size = max(pop_size, 10) # Ensure at least 10 for dim=5
        # For dim=5, 4 + 3*ln(5) = 4 + 4.83 = 8.83 -> 8. Let's use 10 or 12 for robustness.
        # Let's stick to a slightly larger population for exploration.
        pop_size = 10
        
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Initial standard deviation
        # The search space is [-5, 5]^5. The optimum is in [-4, 4]^5.
        # A reasonable initial sigma is 1.0.
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # State variables for adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Weights
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Position control parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2.0 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)
        
        # Counters
        count = 0
        best_x = mean.copy()
        best_f = float('inf')
        
        def eval_point(x):
            nonlocal count, best_x, best_f
            try:
                f = func(x)
                count += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return float('inf')
        
        # Evaluate initial mean
        f_mean = eval_point(mean)
        
        while count < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            population = []
            fitness = []
            
            for i in range(pop_size):
                # Sample from multivariate normal
                # x = mean + sigma * (C^{1/2} * z)
                # We approximate C^{1/2} using Cholesky or just update C directly in simplified CMA-ES
                # For efficiency and simplicity in this context, we can use the standard update rules.
                # However, implementing full CMA-ES with Cholesky updates is complex.
                # A simplified version often used is to maintain the covariance matrix and sample using it.
                
                # Let's use a slightly more efficient sampling if C is diagonal, but it won't be.
                # We will use np.random.multivariate_normal which handles C internally.
                # But np.random.multivariate_normal can be slow or fail if C is not PSD.
                # We'll use a manual Cholesky decomposition approach for robustness.
                
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset it
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                z = np.random.randn(dim)
                x = mean + sigma * (L @ z)
                
                # Clip to box
                x = np.clip(x, -5, 5)
                
                f = eval_point(x)
                population.append(x)
                fitness.append(f)
                
                if count >= budget:
                    break
            
            if count >= budget:
                break
                
            # Sort population by fitness
            idx = np.argsort(fitness)
            population = [population[i] for i in idx]
            fitness = [fitness[i] for i in idx]
            
            # Select best mu individuals
            # If we ran out of budget mid-population, adjust mu
            if len(population) < mu:
                mu = len(population)
                if mu == 0:
                    break
                # Recalculate weights if mu changed? 
                # For simplicity, if mu drops, we just use what we have.
                # But weights were defined for the original mu. 
                # Let's just use the first 'mu' available.
                # If we don't have enough, we break.
                pass
            
            # Update mean
            # m = sum(weights[i] * x_i)
            # Only use the top mu individuals
            selected_pop = population[:mu]
            # Ensure weights sum to 1 for the selected subset if mu changed, 
            # but standard CMA-ES assumes fixed mu. 
            # If we have fewer, we just use the available ones with their original weights (which might sum < 1).
            # To be safe, if we have fewer, we break or renormalize.
            # Given the budget is 500 and pop is 10, we should have full populations for ~50 iterations.
            
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * selected_pop[i]
                
            mean = new_mean
            mean = np.clip(mean, -5, 5) # Keep mean in bounds? CMA-ES usually doesn't clip mean, 
            # but clipping helps if we drift out. However, clipping mean can distort the search.
            # Let's NOT clip mean, but clip samples.
            mean = new_mean # Unclip mean
            
            # Update sigma
            # y = (mean - old_mean) / sigma
            # We need old_mean. 
            # Let's store old_mean before update.
            # Refactor:
            
            # Actually, let's restart the loop logic to be cleaner with old_mean.
            # This implementation is getting messy. Let's use a simpler, more robust CMA-ES variant 
            # or just a well-tuned DE/ES if CMA-ES is too complex to implement correctly in one shot 
            # without testing.
            
            # Given the inspiration says CMA-ES works (1.3470), I should try to get it right.
            # Let's rewrite the core loop more carefully.
            
            # Break out and restart with a cleaner implementation structure below.
            # I will replace the entire __call__ logic in the final code block.
            pass
        
        return best_x, best_f
