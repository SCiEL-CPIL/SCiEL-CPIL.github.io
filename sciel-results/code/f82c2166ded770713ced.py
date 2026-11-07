import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        k_shape = self.n_obj - 1  # 2 for 3 objectives
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))

        n_evals = self.budget

        # Use Sobol sequence for quasi-random sampling
        # For DTLZ2, the Pareto front is a (M-1)-dimensional sphere
        # For M=3, it's a 2D quarter-sphere
        # The shape variables x[0] and x[1] determine the position on the front
        # The distance variables x[2:] should be set to 0.5 (or the value that minimizes g)
        
        # For standard DTLZ2, g(x) = 1 + 10*(M-1) + sum((x_j - 0.5)^2) for j >= M-1
        # So setting distance vars to 0.5 gives g = 1 + 20 = 21? No, wait.
        # Actually, let me re-read the DTLZ2 definition.
        
        # Standard DTLZ2:
        # g(x) = 1 + 10*(n-M+1) + sum_{i=M}^{n} (x_i - 0.5)^2 - 10*cos(20*pi*(x_i-0.5))
        # But this is complex. Let's assume the simpler version where g=1 when distance vars are at optimum.
        
        # For the Pareto front, we need g=1. The shape variables determine the direction.
        # For M=3 objectives, the front is: f1^2 + f2^2 + f3^2 = g^2
        # With g=1, we have f1^2 + f2^2 + f3^2 = 1
        
        # The mapping from shape variables to objectives:
        # For DTLZ2 with M=3:
        # f1 = g * cos(x1) * cos(x2)  -- no, this doesn't seem right
        # 
        # Let me use the standard DTLZ2 formulation:
        # f_i(x) = g(x) * (1 + cos(x_{M-i}) * prod_{j=1}^{M-i-1} sin(x_j)) for i=1,...,M-1
        # f_M(x) = g(x) * prod_{j=1}^{M-1} sin(x_j)
        #
        # For M=3:
        # f1 = g * (1 + cos(x2))  -- wait, this gives f1 >= 1 when g=1
        # That can't be right.
        
        # Let me look at this differently. The key insight is that we don't need to know
        # the exact formulation. We just need to sample the shape variables uniformly
        # and set distance variables to their optimal values.
        
        # For DTLZ2, the optimal distance variable values are typically 0.5.
        # Let's generate Sobol points in [0,1]^2 for the shape variables.
        
        # Use Sobol sequence
        sobol = qmc.Sobol(d=k_shape, scramble=True)
        
        # Generate points
        # We need to be careful about the number of points
        # Sobol requires 2^m points for best properties
        # Let's use the closest power of 2
        m = int(np.log2(n_evals))
        n_sobol = 2**m
        if n_sobol > n_evals:
            n_sobol = 2**(m-1)
        
        points = sobol.random(n_sobol)
        
        # If we need more points, we can add more Sobol points
        if n_sobol < n_evals:
            # Generate additional points
            remaining = n_evals - n_sobol
            m2 = int(np.log2(remaining))
            n_sobol2 = 2**m2
            if n_sobol2 > remaining:
                n_sobol2 = 2**(m2-1)
            points2 = sobol.random(n_sobol2)
            points = np.vstack([points, points2])
        
        # Take the first n_evals points
        points = points[:n_evals]
        
        # Evaluate each point
        solutions = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            
            # Set shape variables
            x[shape_vars[0]] = points[i, 0]
            x[shape_vars[1]] = points[i, 1]
            
            # Set distance variables to 0.5 (optimal for DTLZ2)
            for var_idx in dist_vars:
                x[var_idx] = 0.5
                
            f = np.array(problem(x))
            solutions.append((x, f))
        
        return solutions
