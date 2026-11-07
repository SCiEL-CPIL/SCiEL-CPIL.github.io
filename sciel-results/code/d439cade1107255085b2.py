import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # Phase 1: Broad exploration with more points to find multiple basins
        n_init = 30
        candidates = []
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            candidates.append((f, x))
            
        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Exploitation of top candidates with (1+1)-ES
        top_k = 5
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            if i >= len(candidates):
                break
            f_start, x_start = candidates[i]
            
            # Local (1+1)-ES
            x = x_start.copy()
            f_x = f_start
            sigma = 0.5
            window = 10
            successes = 0
            stagnation = 0
            max_stagnation = 30
            local_budget = 80 # Max calls per local search
            
            while calls < budget and time.time() < deadline:
                if local_budget <= 0:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                local_budget -= 1
                
                if f_y < f_x:
                    x = y
                    f_x = f_y
                    successes += 1
                    stagnation = 0
                else:
                    stagnation += 1
                    
                if successes % window == 0:
                    sr = successes / window
                    if sr > 0.5:
                        sigma *= 1.2
                    elif sr < 0.3:
                        sigma *= 0.8
                    sigma = np.clip(sigma, 0.01, 5.0)
                    successes = 0
                    
                if stagnation > max_stagnation:
                    break

        # Phase 3: Final Polish around the global best
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.05
            for _ in range(50):
                if calls >= budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                if f_y < f_x:
                    x = y
                    f_x = f_y
                    sigma *= 0.9
                else:
                    sigma *= 0.95
                sigma = max(sigma, 0.001)

        return best_x, best_f
