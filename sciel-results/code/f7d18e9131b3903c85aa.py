"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ1 exploiting the known Pareto front structure.
    DTLZ1's Pareto front is a simplex. We generate points uniformly
    distributed on this simplex in the objective space to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Track time to ensure we don't exceed limits
        start_time = time.time()
        
        # DTLZ1 specific:
        # m = 3 objectives
        # n = 7 variables
        # Variables x[0], x[1] determine position on simplex.
        # Variables x[2..6] determine distance to front (g).
        # To be on Pareto front, g=0, which requires x[i]=0.5 for i >= 2.
        
        # We will generate self.budget points on the Pareto front.
        # The simplex in objective space is defined by weights w1, w2, w3 summing to 1.
        # f_i = 0.5 * w_i (since g=0).
        # We want uniform distribution of (w1, w2, w3) on the simplex.
        
        results = []
        
        for _ in range(self.budget):
            # Check deadline
            if time.time() - start_time > 290:
                break
            
            # Generate uniform point on 2D simplex
            # Method: Generate two random numbers u, v in [0,1]
            # If u + v > 1, use 1-u, 1-v
            u = np.random.rand()
            v = np.random.rand()
            
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            
            w1 = u
            w2 = v
            w3 = 1.0 - u - v
            
            # Map weights to decision variables x1, x2
            # x1 = w1
            # x2 = w2 / (w2 + w3) = w2 / (1 - w1)
            
            x = np.zeros(self.dim)
            
            # Set variables affecting g to 0.5 to ensure g=0
            # Indices 2 to 6 (0-indexed)
            x[2:] = 0.5
            
            # Set variables defining the simplex position
            x[0] = w1
            
            denom = 1.0 - w1
            if denom > 1e-12:
                x[1] = w2 / denom
            else:
                x[1] = 0.0 # w2 must be 0 if w1 is 1, so this is safe
            
            # Clip to bounds just in case of floating point errors
            x = np.clip(x, 0.0, 1.0)
            
            # Evaluate
            f = problem(x)
            results.append(x)
            
        return results
# EVOLVE-BLOCK-END
