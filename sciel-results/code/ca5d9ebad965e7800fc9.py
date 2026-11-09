"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Hybrid optimizer:
    1. Evaluates all 2^d vertices of the box to handle linear/unimodal boundary optima.
    2. Runs a (1+1)-ES from the best vertex to refine if the function is not strictly linear 
       or if the optimum is slightly off-vertex due to specific problem variations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # 1. Generate all 2^dim vertices of the box [-5, 5]^dim
        # For dim=5, this is 32 points.
        # We create a grid of -5 and 5 for each dimension.
        vertices = []
        for i in range(2**self.dim):
            v = []
            for j in range(self.dim):
                # Bit j of i determines if it's -5 or 5
                if (i >> j) & 1:
                    v.append(5.0)
                else:
                    v.append(-5.0)
            vertices.append(np.array(v))
        
        # Shuffle vertices to find good solutions early (improves AOCC)
        rng.shuffle(vertices)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Evaluate vertices
        # We limit this to min(len(vertices), self.budget) to be safe, 
        # though 32 < 500 always for dim=5.
        for v in vertices:
            if calls_made >= self.budget:
                break
            fv = func(v)
            calls_made += 1
            if fv < best_f:
                best_f = fv
                best_x = v
        
        # 2. Run (1+1)-ES from the best vertex found
        # This handles cases where the function might have a local minimum 
        # not at a vertex (e.g. if it's actually quadratic with a linear term, 
        # or if the "linear" description is approximate).
        
        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1

        # Initialize ES parameters
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # Start with a smaller sigma since we are already at a good point
        
        # Run ES for remaining budget
        while calls_made < self.budget:
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2  # Increase step size on success
            else:
                sigma *= 0.9  # Decrease step size on failure
                
            # Safety break if sigma becomes too small
            if sigma < 1e-6:
                break

        return x, fx
# EVOLVE-BLOCK-END
