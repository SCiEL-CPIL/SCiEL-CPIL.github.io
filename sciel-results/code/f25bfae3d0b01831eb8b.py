import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        # Phase 1: Global Exploration
        # Use a smaller number of points to save budget for the effective ES phase
        # The simple ES is very effective, so we just need to get into the basin of attraction
        n_explore = min(20, self.budget // 10) 
        points = rng.uniform(-5, 5, (n_explore, self.dim))
        values = np.array([func(p) for p in points])
        
        if time.time() > deadline:
            best_idx = np.argmin(values)
            return points[best_idx], values[best_idx]

        # Select top 3 points to balance exploration and exploitation
        top_k = 3
        top_indices = np.argsort(values)[:top_k]
        top_points = points[top_indices]
        top_values = values[top_indices]
        
        # Phase 2: Local Exploitation with (1+8)-ES
        # (1+8)-ES allows better exploration of the sharp ridge landscape
        # Adaptive step size targeting ~20% success rate
        remaining_budget = self.budget - n_explore
        lambda_ = 8  # number of offspring per generation
        
        best_x = top_points[0].copy()
        best_f = top_values[0]
        
        # Budget per ES run
        budget_per_es = remaining_budget // top_k
        
        for i in range(top_k):
            x = top_points[i].copy()
            fx = top_values[i]
            # Start with a moderate sigma
            sigma = 1.0
            
            # Run (1+8)-ES for this instance
            gen = 0
            while gen < budget_per_es // lambda_:
                if time.time() > deadline:
                    break
                
                # Generate lambda_ offspring
                offspring = x + sigma * rng.standard_normal((lambda_, self.dim))
                offspring = np.clip(offspring, -5, 5)
                
                # Evaluate all offspring
                off_values = np.array([func(y) for y in offspring])
                
                # Find best offspring
                best_off_idx = np.argmin(off_values)
                best_off_y = offspring[best_off_idx]
                best_off_f = off_values[best_off_idx]
                
                # Update sigma based on success rate
                n_success = np.sum(off_values < fx)
                success_rate = n_success / lambda_
                
                if best_off_f < fx:
                    x = best_off_y.copy()
                    fx = best_off_f
                    # Increase sigma if success rate is high
                    if success_rate > 0.5:
                        sigma *= 1.5
                    elif success_rate > 0.3:
                        sigma *= 1.2
                    else:
                        sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Decrease sigma if success rate is low
                    if success_rate < 0.2:
                        sigma *= 0.7
                    elif success_rate < 0.4:
                        sigma *= 0.85
                    else:
                        sigma *= 0.95
                    # Prevent sigma from becoming too small
                    if sigma < 1e-8:
                        break
                
                gen += 1
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
