import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def counting_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return np.inf
            eval_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)
        
        # Hierarchical Grid Search Strategy
        # Katsuura function has a specific structure: 
        # f(x) = 10/dim * sum(1 - cos(2*pi*(x_i + 1))) / i + sum(|x_i|^0.8 * sin(50*x_i^3 + x_i))
        # The first term is periodic with period 1. The second term is highly oscillatory.
        # A grid search that respects this periodicity can be very effective.
        
        # Phase 1: Coarse Grid Search
        # Evaluate a 3^5 = 243 point grid over [-5, 5]^5
        # This is well within the budget of 500.
        
        # We will use a nested loop to generate grid points.
        # To be efficient, we generate the grid for the first few dimensions and loop.
        
        # 3^5 = 243 points.
        # Let's use 5^5 = 3125 points? No, budget is 500.
        # 3^5 = 243 is a good start.
        # 4^5 = 1024 is too many.
        
        # So, start with 3^5.
        # Then refine the best sub-box with a finer grid.
        
        # The Katsuura function's global minimum is typically near the origin or specific points.
        # Due to the hidden shift/rotation, we need to search the whole box.
        
        # Let's do a 3x3x3x3x3 grid.
        grid_size = 3
        # Points: -5, 0, 5
        # Actually, the function is periodic with period 1 in the cosine term.
        # But the absolute value term |x|^0.8 is not periodic.
        # However, the oscillatory term sin(50*x^3+x) has a frequency that increases with x.
        
        # Let's just do a uniform grid first.
        
        # Generate grid points
        # For dim=5, 3^5 = 243 points.
        
        # We will create a list of all combinations of grid coordinates.
        # To save memory and time, we generate them on the fly or use a simple product.
        
        # Since 243 is small, we can create the full array.
        grid_coords = np.linspace(-5.0, 5.0, grid_size)
        
        # Generate all combinations
        # Using itertools.product would be clean, but let's use numpy meshgrid for speed if needed.
        # For 243 points, even a python loop is fast.
        
        points = []
        for d in range(dim):
            # This is getting complex to do manually for arbitrary dim.
            # Let's use scipy.ndindex or a recursive generator.
            pass
            
        # Simpler approach:
        # Create a list of all grid points.
        # 3^5 = 243.
        
        # Let's use a simple recursive function or just hardcode for dim=5?
        # The task says dim=5.
        
        # We'll generate the points using a product of the grid coordinates.
        import itertools
        
        all_points = list(itertools.product(grid_coords, repeat=dim))
        
        # Shuffle the points? 
        # The scoring metric AOCC rewards early progress.
        # If we evaluate in a random order, we might find a good point early by chance.
        # If we evaluate in a structured order (e.g., center first), we might find the global min faster if it's near the center.
        # Given the hidden shift, the global min could be anywhere.
        # However, Katsuura's global minimum is often near 0.
        # Let's prioritize points closer to the center (0,0,0,0,0).
        
        # Sort points by distance to origin
        all_points = np.array(all_points)
        distances = np.linalg.norm(all_points, axis=1)
        sorted_indices = np.argsort(distances)
        all_points = all_points[sorted_indices]
        
        # Evaluate the grid
        for p in all_points:
            if eval_count >= budget or time.time() > deadline:
                break
            f_val = counting_func(p)
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
                
        # Phase 2: Refinement
        # If we have budget left, refine the best region.
        # The best point found in the coarse grid is likely in a box of size (10/2) = 5? No, grid size 3 means step size 5.
        # Box around best_x with half-width = step_size/2 = 2.5.
        
        if eval_count < budget and best_x is not None and time.time() < deadline:
            remaining = budget - eval_count
            if remaining > 20:
                # Refine with a finer grid around best_x
                # Let's use a 5x5x5x5x5 grid? 5^5 = 3125. Too many.
                # Use 3x3x3x3x3 again but with a smaller box?
                # Or use CMA-ES from this point?
                # The prompt suggests "recursively refining the best sub-box with progressively finer grids".
                
                # Let's try a 5-point grid in each dimension? 5^5 is too big.
                # How about a 3-point grid but in a smaller box?
                # Box size: original step was 5. New step: 1?
                # So a 3-point grid with step 1 covers a width of 2.
                
                # Let's define a local box around best_x.
                # Half-width = 1.0
                local_half_width = 1.0
                local_grid_coords = np.linspace(best_x - local_half_width, best_x + local_half_width, 3)
                
                local_points = list(itertools.product(local_grid_coords, repeat=dim))
                local_points = np.array(local_points)
                
                # Sort by distance to best_x
                local_dists = np.linalg.norm(local_points - best_x, axis=1)
                local_sorted = local_points[np.argsort(local_dists)]
                
                for p in local_sorted:
                    if eval_count >= budget or time.time() > deadline:
                        break
                    f_val = counting_func(p)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = p.copy()
                        
                # If we still have budget, do another level of refinement?
                # Or switch to a local optimizer like Nelder-Mead or CMA-ES for the final polish?
                # The prompt says "without stochastic variance". So maybe stick to grids.
                # But a final high-precision polish pass is suggested in the directive.
                
                if eval_count < budget and best_x is not None and time.time() < deadline:
                    remaining = budget - eval_count
                    if remaining > 10:
                        # Final polish with CMA-ES (stochastic, but very effective for final convergence)
                        # The prompt said "without stochastic variance" for the *grid* part, but a final polish is a standard technique.
                        # However, the directive says "Do NOT switch to a different design family".
                        # Is CMA-ES a different family? The current program uses CMA-ES.
                        # So using it for polish is consistent.
                        
                        try:
                            res_polish = scipy.optimize.cma(
                                counting_func,
                                best_x,
                                sigma0=0.1,
                                options={
                                    'maxiter': max(1, remaining // 8),
                                    'popsize': 8,
                                    'tol': 1e-9,
                                    'verbose': 0
                                }
                            )
                            if res_polish.fun < best_f:
                                best_f = res_polish.fun
                                best_x = res_polish.x.copy()
                        except Exception:
                            pass
        
        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            if eval_count < budget:
                best_f = counting_func(best_x)
            else:
                best_f = func(best_x)
                eval_count += 1

        return best_x, best_f
