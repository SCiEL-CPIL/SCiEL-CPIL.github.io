"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Coordinate Descent (DFCD) with multi-start and adaptive steps
    tailored for the bi-modal Lunacek bi-Rastrigin function f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = float('inf')
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made, best_x, best_fx
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Multi-start points to handle bi-modality
        # The Lunacek function has a global minimum and a local minimum.
        # Starting from diverse points helps find the global one.
        starts = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.array([3.0, -3.0, 3.0, -3.0, 3.0][:dim]),
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0][:dim])
        ]
        
        # Filter starts to ensure they are within bounds (though they are)
        # and unique.
        
        # Step sizes for coordinate descent
        # We use a geometric progression.
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
        
        # Allocate budget roughly equally among starts
        # But we can stop early if we run out of budget.
        
        for start_x in starts:
            if calls_made >= budget - 10:
                break
            
            x = start_x.copy()
            fx = eval_point(x)
            
            # If this start is already worse than the current best by a lot,
            # we might skip detailed descent, but usually it's better to try.
            # However, to save budget, if fx is much worse than best_fx, 
            # we might reduce the number of step sizes we try.
            
            # Perform coordinate descent
            for step in step_sizes:
                if calls_made >= budget - 10:
                    break
                
                improved = False
                # One pass over all dimensions for this step size
                for i in range(dim):
                    if calls_made >= budget - 10:
                        break
                    
                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] = min(5.0, x_new[i] + step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            # If we improved, we might want to check negative direction
                            # or just continue to next dim.
                            # Standard CD: move to new point, then next dim.
                            continue
                    
                    # Try negative direction
                    x_new = x.copy()
                    x_new[i] = max(-5.0, x_new[i] - step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                
                # If no improvement in this step size, we could break early 
                # if step is very small, but let's let the loop finish 
                # as smaller steps might still help if we missed a valley.
                # However, to save budget, if step < 0.01 and no improvement, break.
                if step < 0.01 and not improved:
                    break

        # Final local refinement around the best point found
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_fx
            
            # Very fine steps for final polish
            fine_steps = [0.001, 0.0005, 0.0002, 0.0001]
            for step in fine_steps:
                if calls_made >= budget - 5:
                    break
                
                improved = False
                for i in range(dim):
                    if calls_made >= budget - 5:
                        break
                    
                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] = min(5.0, x_new[i] + step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            continue
                    
                    # Try negative direction
                    x_new = x.copy()
                    x_new[i] = max(-5.0, x_new[i] - step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            
                if step < 0.0005 and not improved:
                    break

        return best_x, best_fx

# EVOLVE-BLOCK-END
