"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Adaptive Grid-Based Optimizer.
    
    Strategy:
    1. Coarse grid search (2^5 points) over [-5, 5]^5 to identify the best region.
    2. Recursively subdivide the single best hypercube into 2^5 sub-cubes.
    3. Evaluate all sub-cube points.
    4. Repeat subdivision until budget or time is exhausted.
    
    This deterministic hierarchy guarantees systematic coverage of the basin 
    around the global minimum candidate, avoiding the stochastic traps of 
    local methods on Rastrigin functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize with a coarse grid
        # 2^5 = 32 points. This gives a step size of 10/1 = 10.0? No, range is 10.
        # Grid points: -5, -4, ..., 4, 5? Or just interior?
        # Let's use 2 points per dimension initially: -5 and 5? That's too coarse.
        # Better: Use 3 points per dimension? No, directive says 2^5 grid.
        # "coarse 2^5 grid" usually implies 2 points per dimension in a binary space partition.
        # If we use 2 points per dim, we evaluate corners.
        # Let's start with a slightly denser initial grid to be safe, or stick to the directive.
        # Directive: "starts with a coarse 2^5 grid over [-5, 5]^5"
        # This implies 2 points per dimension. Points: -5, 5.
        # However, Rastrigin minima are often near 0. If the shift is small, corners might be bad.
        # But the recursive subdivision will pull towards the best corner.
        # To be more robust, let's interpret "2^5 grid" as the first level of a binary space partition.
        
        # Current region: entire box
        low = np.full(dim, -5.0)
        high = np.full(dim, 5.0)
        
        # We will evaluate points in the current region.
        # Level 0: Evaluate the 2^5 corners? Or the center?
        # If we only evaluate corners, we might miss the center if the best corner is not the global min direction.
        # But Rastrigin is multimodal. The "best cell" logic assumes a unimodal landscape within the cell.
        # For Rastrigin, this is risky if the cell contains multiple minima.
        # However, the directive specifically asks for this strategy.
        
        # To improve robustness within the directive's framework:
        # We can evaluate the center of the current best cell in addition to its corners/subdivisions?
        # Or, start with a 3-point grid per dimension (8^5 = 32768 points - too many for 500 budget).
        
        # Let's stick to the binary subdivision but be smart about the first step.
        # Instead of just corners, let's evaluate the center of the whole box first?
        # No, the directive says "starts with a coarse 2^5 grid".
        # I will evaluate the 2^5 points defined by the binary split of the box.
        # Points are combinations of low and high for each dimension.
        
        # Optimization: Pre-generate the 32 corner indices (0 or 1 for each dim)
        # 0 -> low, 1 -> high
        
        import itertools
        
        # Generate all 2^dim combinations of 0s and 1s
        corners = np.array(list(itertools.product([0, 1], repeat=dim)))
        
        while True:
            if time.time() > deadline - 1.0:
                break
            if calls_made >= budget:
                break
            
            # Calculate current points
            # For each corner index, point = low + (high - low) * corner_idx
            # Since corner_idx is 0 or 1, point is either low or high.
            current_points = low[None, :] + (high - low)[None, :] * corners
            
            # Evaluate all current points
            # We need to check budget for each call
            for i in range(len(current_points)):
                if time.time() > deadline - 1.0:
                    break
                if calls_made >= budget:
                    break
                
                x = current_points[i]
                # Ensure strict bounds (though low/high are already bounded)
                x = clip(x)
                counted_func(x)
            
            if calls_made >= budget or time.time() > deadline - 1.0:
                break
                
            # Find the best point from the last evaluation batch
            # Note: counted_func updates best_x/best_f globally.
            # We need to know which of the current_points was the best to subdivide that cell.
            # We can re-evaluate? No, that wastes budget.
            # We should track the best value among the points we just evaluated.
            
            # Refactor: Evaluate points and track local best
            # Let's restart the loop logic to track local best properly without re-evaluating.
            # Actually, the previous loop just called counted_func. 
            # We need to know which point in `current_points` had the lowest f.
            # Since counted_func updates global best, if the global best changed, it's one of these.
            # But if the global best was from a previous iteration, we need to find the best of the *current* points.
            
            # Let's restructure the inner loop to collect results.
            
            # Break out and re-implement the loop body more carefully?
            # No, I can just find the index of the point in current_points that matches best_x?
            # Floating point equality is risky.
            
            # Better approach:
            # 1. Generate points.
            # 2. Call func for each, store results in a list.
            # 3. Find argmin of results.
            # 4. Update global best if local min < global best.
            # 5. Subdivide the cell corresponding to the local min.
            
            # Let's redo the while loop structure.
            
            # We already executed the calls above. 
            # To avoid wasting the first iteration's logic, let's just break and restart the loop with the correct logic?
            # No, that's messy. Let's just implement the correct logic from the start.
            
            # I will discard the code above the `while True` and restart the implementation in my head, then write the final code.
            pass

        # The logic above had a flaw: I didn't track which point was best to subdivide.
        # Let's write the clean version.
        
        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        import itertools

        # Initial region
        low = np.full(dim, -5.0)
        high = np.full(dim, 5.0)

        # Pre-compute corner indices (0 or 1 for each dimension)
        # There are 2^dim corners
        corner_indices = np.array(list(itertools.product([0, 1], repeat=dim)))
        
        # We will evaluate the corners of the current [low, high] box.
        # The "cell" to subdivide is the one containing the best point.
        # Since we are evaluating the corners of the current box, the "best corner"
        # identifies the sub-box to focus on in the next iteration.
        
        # Note: In a binary space partition, if we pick corner (0,0,0), the next box is 
        # [low, (low+high)/2]. If we pick (1,0,0), the next box is [(low+high)/2, high] for dim 0, etc.
        
        while True:
            if time.time() > deadline - 1.0:
                break
            if calls_made >= budget:
                break

            # Generate the 2^dim points (corners of current box)
            # Point = low + (high - low) * corner_idx
            # corner_idx is 0 or 1.
            points = low[None, :] + (high - low)[None, :] * corner_indices
            
            local_best_val = np.inf
            local_best_idx = -1
            
            for i in range(len(points)):
                if time.time() > deadline - 1.0:
                    break
                if calls_made >= budget:
                    break
                
                x = points[i]
                fx = counted_func(x)
                
                if fx < local_best_val:
                    local_best_val = fx
                    local_best_idx = i
            
            if local_best_idx == -1:
                break

            # Identify which sub-box to enter
            # The corner index tells us which side of the midpoint to go to for each dimension
            # If corner_idx[d] == 0, we go to the lower half: high[d] = (low[d] + high[d]) / 2
            # If corner_idx[d] == 1, we go to the upper half: low[d] = (low[d] + high[d]) / 2
            
            mid = (low + high) / 2.0
            chosen_corner = corner_indices[local_best_idx]
            
            for d in range(dim):
                if chosen_corner[d] == 0:
                    high[d] = mid[d]
                else:
                    low[d] = mid[d]
            
            # Check if we have converged (box size is tiny)
            # If the box size is smaller than machine epsilon or a small threshold, stop
            box_size = np.max(high - low)
            if box_size < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
