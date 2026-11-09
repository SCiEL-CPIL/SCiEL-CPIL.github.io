"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start coordinate descent for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # Use a fixed seed for reproducibility if needed, but here we use standard rng
        # However, since instances are seeded, we should probably use a fixed strategy 
        # or a rng that doesn't depend on instance seed if we want consistent behavior?
        # The prompt says "hidden seeded shift...". 
        # If I use rng with a fixed seed, my random starts are the same for all instances.
        # This is usually fine for black-box unless the function is adversarially aligned with my seed.
        # Given f14 is "Different powers", it's a standard BBOB function.
        
        rng = np.random.default_rng(42)
        
        # We have 500 budget. 
        # Strategy: 
        # 1. Generate a set of diverse starting points.
        # 2. For each start, run a coordinate descent loop until budget is exhausted or time limit.
        # Coordinate descent is very fast per step (1 func call per dim).
        # To maximize early progress (AOCC), we should do many quick refinements.
        
        # Let's allocate budget to several restarts.
        # If we do 10 restarts, we have ~50 calls each.
        # Or, we can do a "simulated annealing" style coordinate descent with decreasing step size.
        
        # A robust approach for smooth functions: 
        # Start with a few random points.
        # Use a local search that tries to improve each coordinate.
        
        best_x = None
        best_f = float('inf')
        
        # Number of restarts
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        # Generate restart points
        # Mix of center and random
        points = []
        points.append(np.zeros(self.dim))
        for i in range(n_restarts - 1):
            points.append(rng.uniform(-5, 5, self.dim))
        
        # Shuffle to ensure randomness in order
        rng.shuffle(points)
        
        for p in points:
            x = p.copy()
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Coordinate Descent with adaptive step size
            # We will try to improve each coordinate.
            # Step size starts large and decreases.
            
            step_size = 2.0
            # We use remaining budget for this restart
            # But we must check global deadline
            
            # Simple coordinate descent:
            # For each dimension, try x + step and x - step.
            # If improvement, keep and shrink step. Else shrink step.
            
            # To be efficient, we can do a few full passes over coordinates.
            # Each pass costs dim * 2 (or dim * 3) calls.
            # dim=5. Pass cost ~ 10-15 calls.
            
            # Let's do a loop that runs until local budget is used or time is up.
            local_budget = budget_per_restart
            while local_budget > 0:
                if time.time() > deadline:
                    break
                
                improved = False
                for d in range(self.dim):
                    if local_budget < 2:
                        break
                    
                    # Try positive step
                    x_plus = x.copy()
                    x_plus[d] += step_size
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = func(x_plus)
                    local_budget -= 1
                    
                    if f_plus < fx:
                        x = x_plus
                        fx = f_plus
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        improved = True
                        # Keep step size or slightly increase? Usually keep or shrink slightly if we want to converge.
                        # For CD, keeping step size allows moving further.
                        # But to converge, we might need to shrink.
                        # Let's keep step size if improved, but cap it.
                    else:
                        # Try negative step
                        x_minus = x.copy()
                        x_minus[d] -= step_size
                        x_minus = np.clip(x_minus, -5, 5)
                        f_minus = func(x_minus)
                        local_budget -= 1
                        
                        if f_minus < fx:
                            x = x_minus
                            fx = f_minus
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            improved = True
                        else:
                            # No improvement in this coordinate with current step
                            # Shrink step size
                            step_size *= 0.5
                            # If step size is very small, we might be converged
                            if step_size < 1e-4:
                                break
                    # If we improved, we might want to try again with same step to move further?
                    # Standard CD usually does one pass.
                    # Let's do one pass, then shrink step if no overall improvement in the pass.
                
                if not improved:
                    # No coordinate improved in this pass
                    # Shrink step size significantly
                    step_size *= 0.5
                    if step_size < 1e-4:
                        break
                else:
                    # If we improved, we might keep step size for the next pass to continue moving
                    # But to ensure convergence, let's shrink slightly even if improved?
                    # Actually, if we are moving in a direction, we want to keep moving.
                    # However, if we oscillate, we need to shrink.
                    # Let's shrink slightly if improved to be safe, or keep?
                    # Keeping is better for large moves.
                    # Let's keep step_size if improved.
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
