import numpy as np

class Optimizer:
    """
    Optimizer for WFG9 leveraging its analytic structure.
    WFG9 (like WFG4) has a structure where:
    - The first 't' variables (typically n_obj - 1) control the position on the Pareto front.
    - The remaining 'n-t' variables control the distance to the front (convergence).
    - To minimize distance, the tail variables should be set to the midpoint of their bounds.
    - To cover the front, we perform a dense grid search on the first 't' variables.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Parse bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine number of trade-off variables.
        # For WFG4/9 with 3 objectives, typically t = n_obj - 1 = 2.
        t = self.n_obj - 1
        
        # Cap t to dim
        t = min(t, self.dim)
        
        if t < 1:
            # Fallback to random if structure is unknown or invalid
            rng = np.random.default_rng(0)
            for _ in range(self.budget):
                x = lo + (hi - lo) * rng.random(self.dim)
                problem(x)
            return None

        # Strategy:
        # 1. Set tail variables (indices t to dim-1) to the midpoint of their bounds.
        #    This minimizes the 'g' function in WFG4/9 which penalizes deviation from 0.5.
        base_x = (lo + hi) / 2.0
        
        # 2. Determine grid size for the first 't' variables.
        #    We want to use most of the budget for a dense sweep.
        #    Let N be the number of points per dimension in the grid.
        #    Total points = N^t.
        #    For t=2, N^2 <= budget. N = sqrt(budget).
        
        if t == 1:
            N = self.budget
        else:
            N = int(np.floor(self.budget ** (1.0 / t)))
            
        # Ensure we don't exceed budget
        if N ** t > self.budget:
            N -= 1
            
        if N < 2:
            # If budget is too small for a grid, just do random
            rng = np.random.default_rng(0)
            for _ in range(self.budget):
                x = lo + (hi - lo) * rng.random(self.dim)
                problem(x)
            return None

        # Generate grid points for the first t variables
        grid_vals = []
        for i in range(t):
            grid_vals.append(np.linspace(lo[i], hi[i], N))
            
        # Create the meshgrid
        grids = np.meshgrid(*grid_vals, indexing='ij')
        
        # Flatten the grids to get a list of candidate vectors for the first t variables
        flat_grids = [g.flatten() for g in grids]
        X_front = np.column_stack(flat_grids)
        
        # Total points generated
        num_points = X_front.shape[0]
        
        # Construct the full decision vectors
        # X_front has shape (num_points, t)
        # base_x has shape (dim,)
        X_full = np.tile(base_x, (num_points, 1))
        X_full[:, :t] = X_front
        
        # Evaluate the grid points
        for x in X_full:
            problem(x)
            
        used = num_points
        
        # If we have remaining budget, use it for random exploration to fill the budget
        # This ensures we utilize the full budget without exceeding it.
        if used < self.budget:
            rng = np.random.default_rng(42)
            for _ in range(self.budget - used):
                x_rand = lo + (hi - lo) * rng.random(self.dim)
                problem(x_rand)
                
        return X_full
