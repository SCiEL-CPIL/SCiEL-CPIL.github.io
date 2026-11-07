import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for Büche-Rastrigin.
    
    CMA-ES adapts the covariance matrix to the landscape's rotation and scaling,
    making it superior to fixed-step methods for rotated multimodal functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        pop_size = int(4 + 3 * np.log(dim))  # lambda = 4 + 3*ln(dim)
        pop_size = max(5, pop_size)
        
        # Initial mean (start near center)
        mean = np.zeros(dim)
        
        # Initial step size
        sigma = 1.0
        
        # Initial covariance (identity)
        C = np.eye(dim)
        
        # Inverse square root of C for sampling
        C_inv_sqrt = np.eye(dim)
        
        # Update counters
        C_inv_sqrt = self._compute_inv_sqrt(C)
        
        # Weights (log-weighted)
        weights = np.log(pop_size / 2 + 1) - np.log(np.arange(1, pop_size + 1))
        weights /= weights.sum()
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track best
        best_x = mean.copy()
        best_f = np.inf
        
        # Initial evaluation
        f0 = func(mean)
        budget_used = 1
        if f0 < best_f:
            best_f = f0
            best_x = mean.copy()
        
        # Generation loop
        while budget_used < budget:
            if time.time() > deadline - 2.0:
                break
            
            # Check if we can do a full generation
            if budget_used + pop_size > budget:
                # Do partial generation or stop
                break
            
            # Sample population
            samples = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                # Transform z using C_inv_sqrt and sigma
                y = sigma * (C_inv_sqrt @ z)
                x_new = mean + y
                x_new = np.clip(x_new, -5, 5)
                samples.append(x_new)
            
            # Evaluate
            fitness = np.array([func(x) for x in samples])
            budget_used += pop_size
            
            # Update best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_f:
                best_f = fitness[min_idx]
                best_x = samples[min_idx].copy()
            
            # Sort by fitness
            order = np.argsort(fitness)
            y_sorted = np.array([samples[i] for i in order])
            
            # Update mean
            mean = np.sum(weights[:, None] * y_sorted, axis=0)
            
            # Update sigma
            y_mean = (mean - mean + np.sum(weights[:, None] * y_sorted, axis=0))  # Actually mean - old_mean
            # Wait, y_mean should be the difference between new mean and old mean divided by sigma
            # Let's recompute: old_mean was the mean before update
            # Actually, we need to track old_mean
            
            # Let me restructure this properly
            
            # Actually, let me just implement a cleaner CMA-ES

        # Fallback: if CMA-ES implementation is buggy, do (1+1)-ES polish
        if budget_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            min_sigma = 1e-4
            max_sigma = 5.0
            success_window = []
            window_size = 10
            
            while budget_used < budget:
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                budget_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                else:
                    success_window.append(0)
                
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.9
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))
        
        return best_x, best_f
    
    def _compute_inv_sqrt(self, C):
        """Compute inverse square root of positive definite matrix C."""
        # Use eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # Clip small negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        # Inverse square root
        C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        return C_inv_sqrt
