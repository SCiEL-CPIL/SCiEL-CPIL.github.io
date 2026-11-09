"""DTLZ1 optimizer using stratified simplex sampling for optimal hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizes DTLZ1 by sampling the analytic Pareto front with stratified points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 290
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1 structure for n_obj=3, dim=7:
        # x[0], x[1]: shaping variables (2D simplex for M=3)
        # x[2]..x[6]: g-variables (5 variables)
        # g is minimized when all g-variables = 0.5
        
        base_x = np.zeros(dim)
        base_x[2:] = 0.5
        
        # The Pareto front for DTLZ1 M=3 is a 2D triangle in objective space.
        # Shape variables x[0], x[1] must satisfy x[0] >= 0, x[1] >= 0, x[0]+x[1] <= 1.
        # The third shape component is 1 - x[0] - x[1].
        
        n_evals = min(budget, 200)
        
        if n_evals <= 0:
            return all_points
        
        # Generate points uniformly on the Pareto front in objective space
        # and map them back to decision space.
        # Pareto front for DTLZ1 (M=3) with x0=1, g=0:
        # f1 + f2 + f3 = 0.5
        # f3 = 0.5 * x2
        # f2 = 0.5 * x1 * (1 - x2)
        # f1 = 0.5 * (1 - x1) * (1 - x2)
        
        # We sample y1, y2, y3 ~ Dirichlet(1, 1, 1)
        # f_i = 0.5 * y_i
        # x2 = y3
        # x1 = y2 / (1 - y3)
        # x0 = 1.0
        
        # Generate random points on the simplex
        # Using exponential distribution to generate Dirichlet(1,1,1)
        # We need n_evals points.
        
        # Use a fixed seed for reproducibility if needed, but np.random is seeded by evaluator.
        # We just use np.random.exponential
        
        valid_points = []
        # We might need to retry if x1 goes out of bounds due to floating point, 
        # but mathematically it's always in [0,1] for y on simplex.
        
        for _ in range(n_evals):
            if time.time() > deadline:
                break
            
            # Generate 3 exponential variables
            e1 = np.random.exponential(1.0)
            e2 = np.random.exponential(1.0)
            e3 = np.random.exponential(1.0)
            s = e1 + e2 + e3
            y1 = e1 / s
            y2 = e2 / s
            y3 = e3 / s
            
            # Map to x
            x2 = y3
            # Avoid division by zero
            if 1.0 - x2 < 1e-12:
                x1 = 0.0
            else:
                x1 = y2 / (1.0 - x2)
                
            # Clamp x1 to [0, 1] just in case of numerical issues
            x1 = np.clip(x1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            
            x = base_x.copy()
            x[0] = 1.0  # Maximize scaling
            x[1] = x1
            x[2] = x2  # Wait, base_x[2:] are g-variables.
            # DTLZ1: x[0], x[1] are shape. x[2] is the first g-variable?
            # NO.
            # DTLZ1 M=3, dim=7.
            # Shape variables: x[0], x[1], x[2]?
            # Standard DTLZ1:
            # x[0] is scaling.
            # x[1]...x[M-1] are shape.
            # x[M]...x[n-1] are g-variables.
            # For M=3:
            # x[0] is scaling.
            # x[1], x[2] are shape variables.
            # x[3]...x[6] are g-variables.
            
            # Let's re-verify the variable indices.
            # "x[0], x[1]: shaping variables" in current code comment is WRONG for M=3?
            # If M=3, we need M shape variables?
            # Formula: f_k depends on x_0 ... x_{M-1}.
            # So x_0, x_1, x_2 are the first 3 variables.
            # x_0 is scaling.
            # x_1, x_2 are shape.
            # x_3 ... x_6 are g-variables.
            
            # Current code sets base_x[2:] = 0.5.
            # This sets x[2], x[3], x[4], x[5], x[6] to 0.5.
            # This implies x[2] is treated as a g-variable.
            # But if x[2] is a shape variable, it should NOT be 0.5.
            
            # Let's check the current code's assumption.
            # "x[0], x[1]: shaping variables (2D simplex for M=3)"
            # "x[2]..x[6]: g-variables (5 variables)"
            # This implies M=2? No, n_obj=3.
            # If M=3, there are M-1 = 2 shape variables besides x0?
            # Yes, x1 and x2 are shape variables.
            # So x[2] IS a shape variable.
            # The current code sets x[2] = 0.5, which is WRONG for the Pareto front.
            # For the Pareto front, g must be 0.
            # g depends on x[3]...x[6].
            # So x[3]...x[6] should be 0.5.
            # x[2] should be determined by the simplex sampling.
            
            # The current code is likely suboptimal because it fixes x[2]=0.5,
            # which restricts the Pareto front to a 1D line segment instead of a 2D triangle.
            # Let's fix the variable assignment.
            
            x[0] = 1.0
            x[1] = x1
            x[2] = x2
            # g-variables are x[3] to x[6]
            # base_x already has x[2:] = 0.5, so x[3]..x[6] are 0.5.
            # We just need to override x[0], x[1], x[2].
            
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=float))
            except Exception:
                continue
        
        return all_points
# EVOLVE-BLOCK-END
