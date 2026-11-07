import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = 0

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 280  # Safety margin
        
        # IPOP-CMA-ES parameters
        # Start with small population, increase on stagnation
        lam = 4
        sigma = 2.0
        # Initial mean: random in [-5, 5]
        mean = np.random.uniform(-5, 5, self.dim)
        
        # Track progress for restart decision
        best_f_so_far = np.inf
        best_x_so_far = mean
        calls_since_restart = 0
        f_at_restart = np.inf
        
        # CMA-ES internal state
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        mean_prev = mean.copy()
        
        # Weights and parameters will be updated when lam changes
        mu = lam // 2
        weights = self._calc_weights(mu)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        
        # Evaluate initial mean
        f_init = func(mean)
        self.calls += 1
        if f_init < self.best_f:
            self.best_f = f_init
            self.best_x = mean.copy()
            
        best_f_so_far = f_init
        f_at_restart = f_init
        calls_since_restart = 1
        
        while self.calls < self.budget:
            if time.time() > self.deadline:
                break
            
            # Generate population
            z = np.random.randn(lam, self.dim)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
                L = np.eye(self.dim)
                
            pop = mean + sigma * (z @ L.T)
            
            f_vals = []
            for i in range(lam):
                if self.calls >= self.budget or time.time() > self.deadline:
                    break
                x_i = np.clip(pop[i], -5, 5)
                f_i = func(x_i)
                self.calls += 1
                calls_since_restart += 1
                f_vals.append((f_i, x_i))
                
                if f_i < self.best_f:
                    self.best_f = f_i
                    self.best_x = x_i.copy()
                if f_i < best_f_so_far:
                    best_f_so_far = f_i
                    best_x_so_far = x_i.copy()
            
            if len(f_vals) == 0:
                break
                
            f_vals.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.zeros(self.dim)
            for i in range(min(mu, len(f_vals))):
                new_mean += weights[i] * f_vals[i][1]
            mean = new_mean
            
            # Update path variables
            try:
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.eye(self.dim)
                
            # Note: L_inv @ (mean - mean_prev) is the normalized step
            step = L_inv @ (mean - mean_prev)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (step / sigma)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            # h_sigma calculation
            # Using the standard heuristic
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * (calls_since_restart + 1)))
            if denom == 0:
                h_sigma = 0
            else:
                h_sigma = 1 if (norm_p_sigma / denom < (1.4 + 2 / (self.dim + 1)) * chi_n) else 0
                
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (step / sigma)
            
            # C update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            for i in range(min(mu, len(f_vals))):
                y_i = L_inv @ (f_vals[i][1] - mean_prev)
                C += c_mu * weights[i] * np.outer(y_i, y_i)
                
            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            mean_prev = mean.copy()
            
            # Restart Logic (IPOP)
            # Condition 1: No significant improvement in last 'lam * 10' evaluations
            # Condition 2: Sigma is too small (converged locally)
            # Condition 3: Sigma is too large (diverging)
            
            # Simple stagnation check:
            # If best_f_so_far hasn't improved by more than 1% in the last 50 calls
            # Or if sigma < 1e-4
            # Or if sigma > 100
            
            # Let's use a fixed window for stagnation
            # If calls_since_restart > 50 and best_f_so_far > f_at_restart * 0.99 (or similar)
            # Actually, for Lunacek, we want to jump to the other minimum.
            # A common IPOP strategy: restart if lambda * 15 evaluations passed without significant improvement.
            
            if calls_since_restart > lam * 15:
                # Check if we made progress
                # If best_f_so_far is much better than f_at_restart, we are progressing.
                # If not, restart.
                # "Significant" is relative. Let's say if it didn't drop by at least 1e-6 (absolute) 
                # or 1% (relative).
                # Since f_opt can be in [-100, 100], absolute 1e-6 is too small.
                # Let's use relative improvement.
                if f_at_restart - best_f_so_far < 1e-4 * max(1.0, abs(f_at_restart)):
                    # Stagnation detected
                    # Increase population size
                    lam += 2
                    # Recalculate weights and parameters
                    mu = lam // 2
                    weights = self._calc_weights(mu)
                    mu_eff = 1.0 / np.sum(weights**2)
                    
                    c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
                    c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
                    c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
                    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
                    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
                    chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
                    
                    # Reset state
                    C = np.eye(self.dim)
                    p_sigma = np.zeros(self.dim)
                    p_c = np.zeros(self.dim)
                    
                    # Restart mean: random point or current best?
                    # IPOP usually restarts with a random point to explore new regions,
                    # but keeping the best found so far as the center of a larger search
                    # might be safer for AOCC?
                    # Standard IPOP restarts the *mean* at a random point if the previous
                    # run was successful? No, usually it restarts the whole CMA-ES process.
                    # Let's restart mean at a random point to ensure we explore the other minimum.
                    # However, if we found a good local min, we might want to keep it in the population?
                    # Simplest IPOP: Reset mean to random, reset sigma to initial value (or larger).
                    # Let's try: mean = random, sigma = 2.0
                    mean = np.random.uniform(-5, 5, self.dim)
                    mean_prev = mean.copy()
                    sigma = 2.0
                    
                    # Update tracking variables
                    f_init = func(mean)
                    self.calls += 1
                    calls_since_restart = 1
                    if f_init < self.best_f:
                        self.best_f = f_init
                        self.best_x = mean.copy()
                    best_f_so_far = min(best_f_so_far, f_init)
                    f_at_restart = best_f_so_far
                    
                    # If budget is exhausted, break
                    if self.calls >= self.budget:
                        break

        return self.best_x, self.best_f

    def _calc_weights(self, mu):
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        return weights
