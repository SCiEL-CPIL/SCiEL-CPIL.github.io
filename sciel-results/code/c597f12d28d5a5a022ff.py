"""Rewritten program for BBOB f23 (Katsuura) optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # Global best tracking
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1
        
        # (1+1)-ES parameters
        sigma = 1.0
        successes = 0
        steps_in_window = 0
        window_size = 15
        stall_limit = 25  # Restart if no improvement for this many steps
        
        x = x_best.copy()
        fx = f_best
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            # Check for improvement
            if fy < fx:
                x, fx = y, fy
                successes += 1
                stall_count = 0
                if fy < f_best:
                    f_best = fy
                    x_best = x.copy()
            else:
                stall_count = getattr(self, '_stall_count', 0) + 1
                # We need to persist stall_count across iterations in the loop
                # Since we are inside __call__, we can just use a local variable
                # But wait, the loop is local. Let's just use a local variable.
                
            # Correcting: stall_count needs to be local to the loop
            # I will restructure the loop to keep state local.
            
            # Adapt sigma (1/5 success rule)
            steps_in_window += 1
            if steps_in_window >= window_size:
                if successes / window_size > 1/5:
                    sigma *= 1.1
                else:
                    sigma /= 1.1
                successes = 0
                steps_in_window = 0
                
            # Restart logic
            # We need to track stall_count properly.
            # Let's rewrite the loop body to be cleaner.
            
            # Break out to restart if stalled
            if stall_count > stall_limit:
                # Restart from a new random point
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                calls += 1
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
                sigma = 1.0
                successes = 0
                steps_in_window = 0
                stall_count = 0

        return x_best, f_best

# The above code has a bug: stall_count is not defined before use in the else branch 
# and is not properly reset in the main flow if not restarting.
# Let's write a clean version.

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(123)
        
        # Global best
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1
        
        # Local state for (1+1)-ES
        x = x_best.copy()
        fx = f_best
        sigma = 1.0
        successes = 0
        steps_in_window = 0
        window_size = 15
        stall_count = 0
        stall_limit = 30
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                stall_count = 0
                if fy < f_best:
                    f_best = fy
                    x_best = x.copy()
            else:
                stall_count += 1
            
            # Adapt sigma
            steps_in_window += 1
            if steps_in_window >= window_size:
                if successes > window_size / 5.0:
                    sigma *= 1.1
                else:
                    sigma /= 1.1
                successes = 0
                steps_in_window = 0
                
            # Restart if stalled
            if stall_count >= stall_limit:
                # Pick a new random starting point
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                calls += 1
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
                sigma = 1.0
                successes = 0
                steps_in_window = 0
                stall_count = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
