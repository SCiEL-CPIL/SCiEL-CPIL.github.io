"""Model-based optimizer for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for linear functions on a box.
    
    Strategy:
    1. Estimate the gradient using finite differences at the origin with step size 5.
    2. Identify the vertex of the box that minimizes the linear function.
    3. Evaluate that vertex.
    4. If the result is poor (indicating estimation failure or non-linearity), 
       fall back to a (1+1)-ES for the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        d = self.dim
        
        # Phase 1: Gradient Estimation
        # Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_used = 1
        
        # Evaluate at epsilon * e_i
        # Using h=5 allows us to directly probe the boundary vertices along axes
        # This gives a robust estimate of the slope across the entire domain
        h = 5.0
        gradient = np.zeros(d)
        
        for i in range(d):
            if time.time() > deadline - 10:
                break
            x_probe = np.zeros(d)
            x_probe[i] = h
            f_probe = func(x_probe)
            gradient[i] = (f_probe - f0) / h
            calls_used += 1
            
        # Phase 2: Determine Candidate Vertex
        # For a linear function f(x) = c + v^T x, the minimum on [-5, 5]^d 
        # is at the vertex where x_i = -5 if v_i > 0, else 5.
        x_vertex = np.zeros(d)
        for i in range(d):
            if gradient[i] > 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 5.0
                
        f_vertex = func(x_vertex)
        calls_used += 1
        
        # Track best
        best_x = x_vertex
        best_f = f_vertex
        
        # If the vertex is not better than the origin, something is wrong 
        # (e.g. gradient estimation error, or function is not linear).
        # In a valid linear function with non-zero gradient, vertex should be better.
        # If gradient is zero everywhere, f0 is the minimum.
        if f_vertex >= f0:
             # Fall back to local search around origin or just keep origin
             best_x = x0
             best_f = f0
             
        # Phase 3: Refinement / Fallback
        # Since the function is linear, the vertex is the global minimum.
        # We have used ~6 calls. We have budget - 6 calls left.
        # We can use the remaining budget to verify or just do nothing.
        # However, to be robust against any slight non-linearity or noise,
        # we can perform a few (1+1)-ES steps starting from the vertex.
        # But for a pure linear function, moving away from the vertex increases f.
        # So, we should probably just stick with the vertex.
        
        # Let's add a simple check: if the improvement is not significant, 
        # maybe the gradient was estimated poorly?
        # If f_vertex is not significantly smaller than f0, we might try 
        # a different epsilon or restart. But for linear, it should be exact.
        
        # To utilize the budget and potentially handle edge cases (like noise),
        # we will run a (1+1)-ES for the remaining time/budget, 
        # but we will initialize it at the best point found so far.
        # If the function is truly linear, the ES will not improve upon the vertex.
        
        remaining_budget = self.budget - calls_used
        if remaining_budget > 0:
             # Use (1+1)-ES as a safety net
             x = best_x.copy()
             fx = best_f
             sigma = 0.5
             rng = np.random.default_rng(42)
             
             for i in range(remaining_budget):
                 if time.time() > deadline:
                     break
                 y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                 fy = func(y)
                 if fy < fx:
                     x, fx = y, fy
                     sigma *= 1.1
                 else:
                     sigma *= 0.9
                     
                 if fx < best_f:
                     best_f = fx
                     best_x = x.copy()
                     
        return best_x, best_f
# EVOLVE-BLOCK-END
