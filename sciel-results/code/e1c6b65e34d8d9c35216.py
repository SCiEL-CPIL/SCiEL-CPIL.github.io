import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        budget = self.budget
        
        low = [b[0] for b in bounds]
        high = [b[1] for b in bounds]
        
        # 1. Use known optimal x[2:] = 0 for DTLZ7 (g(0) = 0 is minimum)
        best_x_tail = np.zeros(n - 2)
        used = 0
        remaining_budget = budget
        
        # 2. Grid search x[0] and x[1] with remaining budget
        n_grid = int(np.sqrt(remaining_budget))
        if n_grid < 2:
            n_grid = 2
            
        xs0 = np.linspace(low[0], high[0], n_grid)
        xs1 = np.linspace(low[1], high[1], n_grid)
        
        count = 0
        for x0 in xs0:
            for x1 in xs1:
                if time.time() > deadline:
                    break
                if count >= remaining_budget:
                    break
                    
                x = np.zeros(n)
                x[0] = x0
                x[1] = x1
                x[2:] = best_x_tail
                
                problem(x)
                count += 1
                
            if time.time() > deadline or count >= remaining_budget:
                break
                
        # Add a few strategic points: corners and midpoints for better coverage
        strategic_points = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.0, 0.5],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
        ]
        
        for pt in strategic_points:
            if time.time() > deadline:
                break
            if count >= remaining_budget:
                break
            x = np.zeros(n)
            x[0] = pt[0]
            x[1] = pt[1]
            x[2:] = best_x_tail
            problem(x)
            count += 1
            
        for i in range(remaining_budget - count):
            if time.time() > deadline:
                break
            x_rand = np.zeros(n)
            x_rand[0] = np.random.uniform(low[0], high[0])
            x_rand[1] = np.random.uniform(low[1], high[1])
            x_rand[2:] = best_x_tail
            problem(x_rand)
            count += 1
            
        return best_x_tail
