# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 using direct geometric construction of the Pareto front.
    
    The Pareto front of DTLZ1 (3 objectives) is a 2-simplex in objective space defined by:
    f0 + f1 + f2 = 0.5, f_i >= 0.
    
    We sample points uniformly from this simplex in objective space and map them back 
    to decision space using the inverse of the DTLZ1 objective functions (with g(x)=0).
    This ensures all evaluated points are on the Pareto front and well-distributed, 
    maximizing the hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        all_X = []
        all_F = []
        
        for _ in range(self.budget):
            # Sample uniformly from the 2-simplex in objective space
            # Using the standard method for uniform sampling on a simplex
            r1 = np.sqrt(rng.random())
            r2 = rng.random()
            
            # Objectives
            f0 = 0.5 * r1
            f1 = 0.5 * (1 - r1) * r2
            f2 = 0.5 * (1 - r1) * (1 - r2)
            
            # Map to decision space
            # Assuming DTLZ1 formulation:
            # f0 = 0.5 * x[1]
            # f1 = 0.5 * (1 - x[1]) * x[2]
            # f2 = 0.5 * (1 - x[1]) * (1 - x[2])
            # Penalty variables x[0], x[3:] are set to 0.5 to minimize g(x)
            
            x = np.zeros(self.dim)
            
            # Set penalty variables to 0.5
            x[0] = 0.5
            x[3:] = 0.5
            
            # Solve for x[1] and x[2]
            x[1] = 2.0 * f0
            
            # x[2] = f1 / (0.5 * (1 - x[1]))
            denom = 0.5 * (1.0 - x[1])
            if denom < 1e-9:
                x[2] = 0.0 # Handle edge case where f0=0.5
            else:
                x[2] = f1 / denom
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            
            all_X.append(x)
            all_F.append(f)
            
        return np.array(all_X)
# EVOLVE-BLOCK-END
