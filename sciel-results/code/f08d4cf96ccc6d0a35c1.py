"""Adaptive Random Search with Plateau-Escape for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Adaptive Random Search with Plateau-Escape.
    
    The Step Ellipsoidal function (f7) has large flat regions (plateaus) 
    and a sharp drop to the minimum. Standard local searches get stuck on 
    plateaus. This algorithm detects plateaus (no change in function value) 
    and aggressively increases the step size to jump over them. 
    When an improvement is found, it decreases the step size to refine the 
    location.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()

        # Phase 1: Broad Initial Sampling
        # Find a good starting point and estimate the function scale.
        n_initial = 10
        initial_points = []
        for _ in range(n_initial):
            if calls_made >= budget or time_left() < 10.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            initial_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if not initial_points:
            return best_x, best_f
            
        # Estimate a baseline scale for step size
        # If all initial points have similar values, we are likely on a plateau.
        # We start with a moderate step size.
        
        # Phase 2: Multi-Start Adaptive Random Search with Plateau Escape
        # We run multiple independent searches.
        n_restarts = 10
        calls_per_restart = max(20, (budget - calls_made) // n_restarts)
        
        # Sort initial points by fitness to start from good locations
        initial_points.sort(key=lambda p: p[1])
        
        for r in range(n_restarts):
            if calls_made >= budget or time_left() < 5.0:
                break
            
            # Choose starting point
            # Cycle through the best initial points, then random
            if r < len(initial_points):
                x, fx = initial_points[r]
                x = x.copy()
            else:
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Initial step size: 
            # If we are on a plateau, we want to jump. 
            # Start with sigma = 1.0, which is a reasonable fraction of the domain.
            sigma = 1.0
            min_sigma = 0.01
            max_sigma = 10.0
            
            # Plateau detection parameters
            plateau_count = 0
            max_plateau_steps = 10 # If no change for 10 steps, assume plateau
            
            for step in range(calls_per_restart):
                if calls_made >= budget or time_left() < 3.0:
                    break
                
                # Generate candidate
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                # Check for improvement
                if fy < fx:
                    # Accept
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    # Decrease step size to refine
                    sigma *= 0.8
                    plateau_count = 0
                    
                elif np.isclose(fy, fx, rtol=1e-5, atol=1e-8):
                    # Plateau detected: no change in function value
                    # Aggressively increase step size to escape the flat region
                    sigma *= 1.5
                    plateau_count += 1
                    
                    # If we've been on the plateau for a while, 
                    # we might want to restart from a different location 
                    # or just keep jumping. 
                    # For now, keep jumping.
                    
                else:
                    # Rejection (fy > fx)
                    # Decrease step size slightly
                    sigma *= 0.9
                    plateau_count = 0
                
                # Clamp step size
                sigma = np.clip(sigma, min_sigma, max_sigma)
                
                # If sigma becomes very small and we are not improving, 
                # we are stuck in a local minimum or a small plateau.
                # We could trigger a restart here, but the outer loop handles restarts.
        
        # Phase 3: Final Polish
        # Use the best point found and do a fine-grained local search
        if best_x is not None and calls_made < budget and time_left() > 5.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_steps = min(50, budget - calls_made)
            
            for _ in range(polish_steps):
                if calls_made >= budget or time_left() < 2.0:
                    break
                
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 0.9 # Slowly converge
                else:
                    sigma *= 0.8 # Reduce step on failure
                
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
