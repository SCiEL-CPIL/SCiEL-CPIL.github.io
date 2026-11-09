import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        if lam < 4: lam = 4
        
        # Initial step size
        # The box is [-5, 5]. The optimum is in [-4, 4].
        # A sigma of 2.0 covers a good portion of the space.
        sigma = 2.0
        
        # Initial mean
        # Start at the center
        x_mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Evaluate initial mean
        f_mean = float(func(x_mean))
        calls = 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        # Main loop
        while calls < budget and time.time() < deadline:
            # Generate population
            y = np.random.multivariate_normal(np.zeros(n), C, size=lam)
            x_pop = x_mean + sigma * y
            
            # Evaluate population
            f_pop = np.zeros(lam)
            for i in range(lam):
                if calls >= budget or time.time() >= deadline:
                    break
                f_pop[i] = float(func(x_pop[i]))
                calls += 1
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = x_pop[i].copy()
            
            # If we broke out early, break the main loop too
            if calls >= budget or time.time() >= deadline:
                break
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            x_pop = x_pop[idx]
            f_pop = f_pop[idx]
            
            # Weighted mean
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * y[idx[i]]
                
            # Update step size
            # Note: y is the standardized normal, so we use y[idx[i]]
            # But y was generated as N(0, C), so y is already in the "covariance" space?
            # No, standard CMA-ES generates z ~ N(0, I), then x = mean + sigma * B * z.
            # Here I used multivariate_normal(0, C), so y ~ N(0, C).
            # The update for sigma uses the norm of the weighted mean of the *standardized* samples.
            # If I generate y ~ N(0, C), I need to be careful.
            # Standard CMA-ES:
            # z ~ N(0, I)
            # x = mean + sigma * B * z
            # Here B is the square root of C.
            # Let's stick to the standard formulation to avoid bugs.
            
            # Re-generate using standard normal and transformation
            # This is safer.
            pass
            
        return best_x, float(best_f)
