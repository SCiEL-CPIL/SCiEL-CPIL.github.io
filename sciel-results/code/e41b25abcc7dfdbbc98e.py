import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        
        # Time guard
        start_time = time.time()
        time_limit = 290.0
        
        def safe_eval(x):
            if self.calls >= self.budget:
                return self.best_f
            if time.time() - start_time > time_limit:
                return self.best_f
            self.calls += 1
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Scout (50 calls)
        scout_budget = min(50, budget)
        best_x_scout = None
        best_f_scout = np.inf
        
        for i in range(scout_budget):
            if time.time() - start_time > time_limit:
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_eval(x)
            if f < best_f_scout:
                best_f_scout = f
                best_x_scout = x.copy()
        
        if best_x_scout is None:
            best_x_scout = np.zeros(dim)
            best_f_scout = safe_eval(best_x_scout)
            
        # Phase 2: (1+10)-ES with Restarts
        current_x = best_x_scout.copy()
        current_f = best_f_scout
        sigma = 1.0
        lambda_ = 10
        stagnation = 0
        max_stagnation = 50
        
        while self.calls < budget and time.time() - start_time < time_limit:
            if budget - self.calls < lambda_:
                break
            
            best_offspring_x = current_x
            best_offspring_f = current_f
            improved = False
            
            for i in range(lambda_):
                if self.calls >= budget or time.time() - start_time > time_limit:
                    break
                z = rng.standard_normal(dim)
                x = current_x + sigma * z
                f = safe_eval(x)
                
                if f < best_offspring_f:
                    best_offspring_f = f
                    best_offspring_x = x.copy()
                    improved = True
                    
            if self.calls >= budget or time.time() - start_time > time_limit:
                break
                
            if improved:
                current_x = best_offspring_x
                current_f = best_offspring_f
                sigma *= 1.1
                stagnation = 0
            else:
                sigma *= 0.9
                stagnation += 1
                
            if stagnation > max_stagnation:
                current_x = rng.uniform(-5, 5, dim)
                if self.calls < budget and time.time() - start_time < time_limit:
                    current_f = safe_eval(current_x)
                else:
                    break
                sigma = 1.0
                stagnation = 0

        return self.best_x, self.best_f
