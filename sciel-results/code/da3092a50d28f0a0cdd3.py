import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(0)
        
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls_made = 1
        
        # Parameters for 1/5 rule
        # c_sigma = 1/3 for dim=5
        c_sigma = 1.0 / 3.0
        # chi_norm for standard normal
        chi_norm = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        # Window size for success rate calculation
        # Typically 1/5 * c_sigma * dim
        window_size = int(1.0/5.0 * c_sigma * self.dim)
        if window_size < 5: window_size = 5
        
        success_history = []
        
        stagnation_threshold = 20
        polish_budget = 50
        es_budget = self.budget - polish_budget
        
        while calls_made < es_budget:
            if time.time() > deadline:
                break
            
            x = x_best.copy()
            sigma = 2.0
            stagnation_count = 0
            local_best_val = f_best
            local_best_x = x_best.copy()
            
            while calls_made < es_budget:
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < local_best_val:
                    local_best_val = fy
                    local_best_x = y
                    x = y
                    success_history.append(1)
                    stagnation_count = 0
                    
                    if fy < f_best:
                        f_best = fy
                        x_best = y
                else:
                    success_history.append(0)
                    stagnation_count += 1
                
                # Maintain window size
                if len(success_history) > window_size:
                    success_history.pop(0)
                
                # 1/5 Rule Adaptation
                if len(success_history) == window_size:
                    success_rate = np.mean(success_history)
                    # sigma = sigma * exp(c_sigma * (success_rate - 1/5))
                    # Using log-domain or direct multiplication
                    sigma *= np.exp(c_sigma * (success_rate - 0.2))
                
                # Restart conditions
                if stagnation_count >= stagnation_threshold or sigma < 1e-8:
                    break
            
            # If we broke out due to stagnation, the loop restarts from x_best
            # We should probably reset sigma to a larger value on restart?
            # The current code resets sigma = 2.0 at the start of the inner loop.
            # This is good.
        
        # Final Polish
        if calls_made < self.budget and time.time() < deadline:
            try:
                # L-BFGS-B
                result = minimize(
                    func,
                    x_best,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxfun': self.budget - calls_made}
                )
                calls_made += result.nfev
                
                if result.fun < f_best:
                    f_best = result.fun
                    x_best = result.x
            except Exception:
                pass
            
            # If budget remains, try Nelder-Mead for robustness
            if calls_made < self.budget and time.time() < deadline:
                try:
                    result2 = minimize(
                        func,
                        x_best,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'maxfun': self.budget - calls_made}
                    )
                    calls_made += result2.nfev
                    if result2.fun < f_best:
                        f_best = result2.fun
                        x_best = result2.x
                except Exception:
                    pass
        
        return x_best, f_best
