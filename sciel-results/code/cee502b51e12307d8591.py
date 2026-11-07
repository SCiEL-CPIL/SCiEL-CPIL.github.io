"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with coordinate descent polish for Schwefel-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def random_point():
            return rng.uniform(-5.0, 5.0, self.dim)
        
        def evaluate(x):
            nonlocal calls
            fx = func(x)
            calls += 1
            return fx

        # Main loop with restarts
        max_stall = 30 # Restart if no improvement for this many calls
        stall_count = 0
        x = random_point()
        fx = evaluate(x)
        
        if fx < best_f:
            best_f = fx
            best_x = x.copy()
            
        sigma = 2.0
        window_size = 10
        recent_outcomes = []
        
        while calls < self.budget and time.time() < deadline:
            # Generate candidate (1+1)-ES step
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5.0, 5.0)
            fy = evaluate(y)
            
            improved = False
            if fy < fx:
                x = y
                fx = fy
                improved = True
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    stall_count = 0
            else:
                stall_count += 1
                
            recent_outcomes.append(1 if improved else 0)
            if len(recent_outcomes) > window_size:
                recent_outcomes.pop(0)
                
            # Adaptive sigma
            if len(recent_outcomes) == window_size:
                success_rate = sum(recent_outcomes) / window_size
                if success_rate > 0.5:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                sigma = max(0.05, min(sigma, 5.0))
                
            # Restart if stuck
            if stall_count > max_stall:
                x = random_point()
                fx = evaluate(x)
                sigma = 2.0
                recent_outcomes = []
                stall_count = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
            # Periodic Coordinate Descent Polish
            # Do this if we have made some progress recently or are close to budget
            if calls % 20 == 0 and calls < self.budget - 10:
                # Simple CD: try small steps along each axis
                for i in range(self.dim):
                    if calls >= self.budget or time.time() >= deadline:
                        break
                    for step_dir in [-1, 1]:
                        if calls >= self.budget or time.time() >= deadline:
                            break
                        trial = x.copy()
                        # Use a small step relative to current sigma or fixed small
                        step_size = max(sigma * 0.1, 0.01)
                        trial[i] += step_dir * step_size
                        trial[i] = np.clip(trial[i], -5.0, 5.0)
                        ft = evaluate(trial)
                        if ft < fx:
                            x = trial
                            fx = ft
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                                stall_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
