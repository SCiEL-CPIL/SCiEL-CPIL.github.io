import numpy as np
import time


class Optimizer:
    """Population-based ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # State
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls_used = 1
        
        # ES parameters
        mu = 5  # number of parents
        lam = 15  # number of offspring
        sigma = 2.0  # initial step size
        sigma_min = 0.01
        sigma_max = 10.0
        
        # Restart parameters
        max_stagnant = 30  # generations without improvement before restart
        stagnant_count = 0
        best_ever = f_best
        
        # Track calls
        def counted_func(x):
            nonlocal calls_used
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5, 5)
            val = func(x)
            calls_used += 1
            return val
        
        while calls_used < self.budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Generate population
            population = []
            for i in range(lam):
                if i < mu:
                    # Use best as base for some offspring
                    y = x_best + sigma * rng.standard_normal(self.dim)
                else:
                    # Random exploration
                    y = x_best + sigma * 1.5 * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = counted_func(y)
                population.append((fy, y))
                
                if calls_used >= self.budget:
                    break
            
            # Sort by fitness
            population.sort(key=lambda t: t[0])
            
            # Update best
            if population[0][0] < f_best:
                f_best = population[0][0]
                x_best = population[0][1]
                stagnant_count = 0
            else:
                stagnant_count += 1
            
            # Adaptive step size
            # Count improvements among offspring
            improvements = sum(1 for fy, _ in population if fy < f_best + 1e-10)
            success_rate = improvements / len(population)
            
            if success_rate > 0.5:
                sigma = min(sigma * 1.2, sigma_max)
            elif success_rate < 0.2:
                sigma = max(sigma * 0.8, sigma_min)
            
            # Restart if stagnant
            if stagnant_count >= max_stagnant:
                # Restart with new random point
                x_best = rng.uniform(-5, 5, self.dim)
                f_best = counted_func(x_best)
                sigma = 2.0
                stagnant_count = 0
                best_ever = min(best_ever, f_best)
        
        return x_best, f_best
