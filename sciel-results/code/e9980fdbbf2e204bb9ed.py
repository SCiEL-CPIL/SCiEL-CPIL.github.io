import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_fx = float('inf')
        calls_made = 0
        
        # Parameters for the (1+1)-ES with restarts
        # We will use a stall-based restart strategy
        stall_limit = 15
        stall_count = 0
        step_size = 1.0
        min_step_size = 1e-6
        max_step_size = 10.0
        
        # Initial point
        current_x = rng.uniform(-5, 5, size=dim)
        current_fx = float('inf')
        
        # We need to evaluate the initial point
        if calls_made < total_budget and time.time() < deadline:
            current_fx = func(current_x)
            calls_made += 1
            best_x = current_x.copy()
            best_fx = current_fx
        
        while calls_made < total_budget and time.time() < deadline:
            # Generate a candidate
            # Simple (1+1)-ES: add Gaussian noise scaled by step_size
            noise = rng.normal(0, step_size, size=dim)
            candidate_x = current_x + noise
            
            # Clip to bounds
            candidate_x = np.clip(candidate_x, -5, 5)
            
            # Evaluate
            if calls_made >= total_budget:
                break
            if time.time() >= deadline:
                break
                
            candidate_fx = func(candidate_x)
            calls_made += 1
            
            if candidate_fx < current_fx:
                # Improvement
                current_x = candidate_x.copy()
                current_fx = candidate_fx
                stall_count = 0
                # Increase step size slightly if we improved
                step_size = min(step_size * 1.2, max_step_size)
                
                if current_fx < best_fx:
                    best_fx = current_fx
                    best_x = current_x.copy()
            else:
                # No improvement
                stall_count += 1
                # Decrease step size
                step_size = max(step_size * 0.8, min_step_size)
                
            # Check for restart
            if stall_count >= stall_limit:
                # Reset to a random point or best point?
                # Let's try a random restart in the box
                # But keep the best found so far
                current_x = rng.uniform(-5, 5, size=dim)
                # Evaluate new current point
                if calls_made < total_budget and time.time() < deadline:
                    current_fx = func(current_x)
                    calls_made += 1
                    if current_fx < best_fx:
                        best_fx = current_fx
                        best_x = current_x.copy()
                else:
                    break
                stall_count = 0
                step_size = 1.0  # Reset step size
                
        return best_x, best_fx
