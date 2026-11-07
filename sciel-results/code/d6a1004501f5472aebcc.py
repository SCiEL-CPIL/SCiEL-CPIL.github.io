import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 280  # 280s safety margin out of 330s
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f
        
        # Phase 1: Quick exploration with diverse starting points
        n_explore = min(5, budget // 10)
        explore_points = [np.zeros(dim)]
        for i in range(n_explore - 1):
            explore_points.append(rng.uniform(-4, 4, dim))
        
        for x0 in explore_points:
            if calls >= budget or time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            f0 = eval_func(x0)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)
        
        # Phase 2: (1+1)-ES with adaptive restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        no_improve = 0
        stuck_threshold = 15
        restart_count = 0
        max_restarts = 10
        
        for iteration in range(budget - calls):
            if time.time() > self.deadline:
                break
            if calls >= budget:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = eval_func(y)
            
            if fy < fx:
                x = y
                fx = fy
                no_improve = 0
                sigma *= 1.3  # Increase step on success
            else:
                no_improve += 1
                sigma *= 0.85  # Decrease step on failure
                
                # If stuck, restart from best-so-far with larger sigma
                if no_improve >= stuck_threshold and restart_count < max_restarts:
                    x = best_x.copy() + sigma * 2.0 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    fx = eval_func(x)
                    sigma = max(sigma, 0.5)
                    no_improve = 0
                    restart_count += 1
                    stuck_threshold = max(10, int(stuck_threshold * 0.8))
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)
        
        return best_x, best_f
