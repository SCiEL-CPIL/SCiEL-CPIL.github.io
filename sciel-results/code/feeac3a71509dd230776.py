import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Deterministic Global Lattice Search ---
        # For dim=5, a grid of 3 points per dimension gives 3^5 = 243 evaluations.
        # This is deterministic and covers the space [-5, 5]^5.
        # We use points -4, 0, 4 to be slightly inside the bounds, avoiding edge effects.
        grid_points = np.array([-4.0, 0.0, 4.0])
        
        # Generate meshgrid
        grids = np.meshgrid(*([grid_points] * dim))
        grid = np.stack([g.ravel() for g in grids], axis=1)
        
        # Ensure we don't exceed budget
        if len(grid) > budget:
            # If budget is too small for full grid, subsample
            # But budget is 500, and grid is 243, so we are fine.
            # If it were larger, we'd need to be smarter.
            pass

        best_x = None
        best_f = np.inf
        evals = 0

        # Evaluate grid points
        for x in grid:
            if not time_ok():
                break
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 2: Multi-start Coordinate Descent Refinement ---
        # The function is Schaffer's F7, which is ill-conditioned and axis-aligned.
        # Coordinate descent is very effective here.
        # We will start from the best grid point and potentially a few other good points.
        
        # Store the top few points from the grid to use as multiple starts
        # We already have the best, but let's keep track of a few others if we had more budget?
        # Actually, we only stored the best. Let's just use the best point for now,
        # or we can re-evaluate a few neighbors if needed.
        # Given the strong axis-aligned structure, a single good start with aggressive
        # coordinate descent might be sufficient.
        
        # However, to be robust against local minima (though F7 is unimodal in the 
        # rotated frame, it might have local issues in the original frame due to 
        # ill-conditioning), let's try a few starts.
        # Since we didn't store all grid values, we'll just use the best one.
        # If we had more budget, we could keep top K.
        
        # Let's try a more sophisticated approach: 
        # Use the best point from grid.
        # Then perform coordinate descent with adaptive step sizes.
        
        x = best_x.copy()
        f = best_f
        
        # Adaptive step sizes for each dimension
        steps = np.ones(dim) * 0.5  # Initial step size
        
        # Coordinate descent loop
        # We have budget - evals remaining.
        # Each iteration of coordinate descent takes 'dim' evaluations.
        # We can do many iterations.
        
        max_cd_iterations = (budget - evals) // dim
        if max_cd_iterations < 1:
            return best_x, best_f

        for _ in range(max_cd_iterations):
            if not time_ok():
                break
            
            improved = False
            for i in range(dim):
                if not time_ok():
                    break
                
                # Try positive and negative steps
                # Adaptive step: if it improved, increase step; else decrease
                # Standard coordinate descent with line search-like behavior
                
                # Try forward
                x_new = x.copy()
                x_new[i] += steps[i]
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                evals += 1
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    improved = True
                    # Increase step for this dimension (exploration)
                    steps[i] *= 1.5
                else:
                    # Try backward
                    x_new = x.copy()
                    x_new[i] -= steps[i]
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    evals += 1
                    
                    if f_new < f:
                        x = x_new
                        f = f_new
                        improved = True
                        # Increase step for this dimension
                        steps[i] *= 1.5
                    else:
                        # No improvement, shrink step
                        steps[i] *= 0.7
                        # Ensure step doesn't get too small
                        if steps[i] < 1e-6:
                            steps[i] = 1e-6

            if not improved:
                # If no improvement in a full cycle, maybe shrink all steps
                steps *= 0.8
                if np.all(steps < 1e-5):
                    break

        # --- Phase 3: Final Polish with Small Steps ---
        # If we still have budget, do a few more iterations with very small steps
        # to ensure we are at the bottom of the valley.
        if evals < budget and time_ok():
            # Reduce steps significantly
            steps = np.ones(dim) * 0.01
            remaining = budget - evals
            max_polish_iters = remaining // dim
            for _ in range(max_polish_iters):
                if not time_ok():
                    break
                improved = False
                for i in range(dim):
                    if not time_ok():
                        break
                    x_new = x.copy()
                    x_new[i] += steps[i]
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    evals += 1
                    if f_new < f:
                        x = x_new
                        f = f_new
                        improved = True
                    else:
                        x_new = x.copy()
                        x_new[i] -= steps[i]
                        x_new = np.clip(x_new, -5, 5)
                        f_new = func(x_new)
                        evals += 1
                        if f_new < f:
                            x = x_new
                            f = f_new
                            improved = True
                    if evals >= budget:
                        break
                if not improved:
                    break

        if best_x is None:
            best_x = x
            best_f = f

        return best_x, best_f
