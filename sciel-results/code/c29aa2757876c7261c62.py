import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse Grid Search
        grid_vals = [-4.0, 0.0, 4.0]
        grid_points = np.array(np.meshgrid(*([grid_vals] * dim))).T.reshape(-1, dim)
        
        eval_count = 0
        best_x = None
        best_f = np.inf
        grid_fits = []
        
        for x in grid_points:
            if eval_count >= budget or time.time() > deadline:
                break
            f_val = func(x)
            eval_count += 1
            grid_fits.append((f_val, x))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
        if eval_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Sort grid results
        grid_fits.sort(key=lambda t: t[0])
        
        # Pick top 5
        K = 5
        top_points = [x for _, x in grid_fits[:K]]
        top_fits = [f for f, _ in grid_fits[:K]]
        
        # Initialize agents
        agents = []
        for i in range(K):
            agents.append({
                'x': top_points[i].copy(),
                'f': top_fits[i],
                'sigma': 1.5
            })
        
        # Main Loop
        while eval_count < budget and time.time() < deadline:
            for agent in agents:
                if eval_count >= budget or time.time() > deadline:
                    break
                
                x = agent['x']
                sigma = agent['sigma']
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                f_y = func(y)
                eval_count += 1
                
                if f_y < agent['f']:
                    agent['x'] = y
                    agent['f'] = f_y
                    agent['sigma'] = min(agent['sigma'] * 1.05, 5.0)
                    
                    if f_y < best_f:
                        best_f = f_y
                        best_x = y.copy()
                else:
                    agent['sigma'] = max(agent['sigma'] * 0.95, 0.01)
                    
                if agent['sigma'] < 0.01:
                    # Restart
                    if rng.random() < 0.5:
                        agent['x'] = best_x + rng.normal(0, 0.5, dim)
                        agent['x'] = np.clip(agent['x'], -5, 5)
                    else:
                        agent['x'] = rng.uniform(-5, 5, dim)
                    
                    agent['f'] = func(agent['x'])
                    eval_count += 1
                    
                    if agent['f'] < best_f:
                        best_f = agent['f']
                        best_x = agent['x'].copy()
                        
                    agent['sigma'] = 1.0

        return best_x, best_f
