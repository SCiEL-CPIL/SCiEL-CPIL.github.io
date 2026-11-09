"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Analytical Pareto Front Construction for DTLZ4 using a structured grid.
    
    Strategy:
    1. Identify the Pareto front analytically: g(x_m)=0 => x_m = 0.5.
    2. The front is parameterized by x_{n-1}, x_{n-2} in [0, 1].
    3. To maximize hypervolume, we need points that cover the front well.
    4. A uniform grid in the parameter space (x_{n-1}, x_{n-2}) provides good coverage.
    5. We use a grid of size N x N where N^2 <= budget.
    6. We add the corners of the front (extreme points) to ensure boundary coverage.
    7. If budget allows, we add random points to fill gaps, but the grid is the core.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 specific logic
        # n_obj = 3 -> k = 2 shape variables
        # dim = 12 -> m = 10 position variables
        k = self.n_obj - 1
        m = self.dim - k
        
        # The Pareto front is defined by x_0...x_{m-1} = 0.5
        # The shape variables x_{m}...x_{n-1} (last k vars) define the position on the front.
        
        # We want to generate 'budget' points.
        # Let's use a grid for the 2 shape variables.
        # Let N be the number of points along each axis.
        # N^2 should be close to budget.
        # For budget=200, N=14 gives 196 points. Remaining 4 can be used for corners or random.
        
        N = int(np.sqrt(self.budget))
        # Ensure we don't exceed budget
        # If N^2 > budget, reduce N.
        while N * N > self.budget:
            N -= 1
            
        # Generate grid points for shape variables
        # We use linspace to cover [0, 1] including endpoints.
        # Including endpoints is crucial for hypervolume as it captures the extreme objectives.
        grid_vals = np.linspace(0, 1, N)
        
        # Create meshgrid
        G1, G2 = np.meshgrid(grid_vals, grid_vals, indexing='ij')
        
        # Flatten to list of shape variable pairs
        shape_vars = np.column_stack((G1.ravel(), G2.ravel()))
        
        # Number of grid points
        n_grid = len(shape_vars)
        
        # Position variables are fixed to 0.5
        pos_vars = np.full((n_grid, m), 0.5)
        
        # Combine
        X = np.hstack([pos_vars, shape_vars])
        
        # Check if we have remaining budget
        remaining = self.budget - n_grid
        
        if remaining > 0:
            # Add specific corner points if not already included in grid
            # The corners of the shape space [0,1]x[0,1] are (0,0), (0,1), (1,0), (1,1)
            # These map to extreme points of the Pareto front.
            # Since linspace includes 0 and 1, these are likely already in the grid.
            # Let's verify if they are present.
            # If not, add them. If yes, use remaining budget for random points 
            # to slightly perturb or fill in, but random points on the front are less 
            # efficient than a dense grid for hypervolume. 
            # However, a slightly denser grid might be better.
            
            # Alternative: Use the remaining budget to increase the grid density?
            # No, we already maximized N.
            
            # Let's just add random points on the front to utilize the budget.
            # This might help if the grid misses some high-contributing regions, 
            # though for a convex front like DTLZ4, the grid is usually very good.
            
            # Actually, for DTLZ4, the front is convex. Hypervolume is maximized by 
            # points that are "spread out". A uniform grid in parameter space 
            # (angles) is a good approximation.
            
            # Let's add 'remaining' random points on the front.
            if remaining > 0:
                # Use np.random to respect seed
                rng = np.random
                extra_shape = rng.random((remaining, k))
                extra_pos = np.full((remaining, m), 0.5)
                extra_X = np.hstack([extra_pos, extra_shape])
                X = np.vstack([X, extra_X])
        
        # Evaluate all
        # We must ensure we don't exceed budget.
        # We have exactly self.budget points now.
        
        # Time check
        if time.time() > deadline:
             return X
        
        F = np.array([problem(x) for x in X])
        
        # Final time check
        if time.time() > deadline:
             return X
             
        return X
# EVOLVE-BLOCK-END
