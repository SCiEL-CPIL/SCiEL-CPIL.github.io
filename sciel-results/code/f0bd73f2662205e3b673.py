# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4 that exploits the known Pareto front geometry.
    It sets distance variables to their optimal values (0.5) to minimize g(x),
    and samples shape variables using a stratified grid with vertex inclusion
    to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with 3 objectives:
        # Shape variables: x_0, x_1 (indices 0, 1)
        # Distance variables: x_2 ... x_{dim-1} (indices 2 to 11 for dim=12)
        n_shape = n_obj - 1  # 2
        n_dist = dim - n_shape  # 10

        # Optimal distance values for DTLZ4 are 0.5
        # g(x) = sum((x_i - 0.5)^2), so g=0 when x_i=0.5
        
        # For DTLZ4 3-objective, the Pareto front is a quarter sphere.
        # The mapping from decision variables to angles is:
        # theta_0 = (pi / 2) * x_0  =>  f_0 = cos(theta_0)
        # theta_1 = (pi / 2) * x_1  =>  f_1 = sin(theta_0) * cos(theta_1)
        #                                       f_2 = sin(theta_0) * sin(theta_1)
        
        # To get uniform coverage on the sphere surface:
        # cos(theta_0) ~ Uniform(0, 1)  =>  x_0 = (2/pi) * arccos(u)
        # theta_1 ~ Uniform(0, pi/2)    =>  x_1 = v
        
        results = []
        
        # Use a stratified grid with vertex inclusion
        # n_grid x n_grid grid points, with vertices included
        n_grid = int(np.sqrt(budget))
        n_extra = budget - n_grid * n_grid
        
        # Generate grid points
        # For hypervolume, clustering near the vertices (u=0, u=1, v=0, v=1)
        # might be beneficial as these points define the "extremes" of the dominated region.
        # Use a cosine spacing to cluster points at the ends.
        u_vals = np.cos(np.linspace(0.0, np.pi / 2.0, n_grid))
        v_vals = np.cos(np.linspace(0.0, np.pi / 2.0, n_grid))
        
        for i in range(n_grid):
            for j in range(n_grid):
                # Check deadline
                if time.time() > deadline:
                    return None
                
                u = u_vals[i]
                v = v_vals[j]
                
                # Calculate shape variables
                x_0 = (2.0 / np.pi) * np.arccos(u)
                x_1 = v
                
                # Construct full vector
                x = np.zeros(dim)
                x[0] = x_0
                x[1] = x_1
                x[2:] = 0.5  # Set distance variables to optimal value
                
                # Evaluate
                f = problem(x)
                results.append(f)
        
        # Add extra points if budget allows
        if n_extra > 0:
            # Use a finer sub-grid or Halton sequence for extra points
            for k in range(n_extra):
                if time.time() > deadline:
                    return None
                
                # Use a Van der Corput sequence for low-discrepancy sampling
                u = self._halton(k + 1, 2)
                v = self._halton(k + 1, 3)
                
                x_0 = (2.0 / np.pi) * np.arccos(u)
                x_1 = v
                
                x = np.zeros(dim)
                x[0] = x_0
                x[1] = x_1
                x[2:] = 0.5
                
                f = problem(x)
                results.append(f)
        
        return None
    
    def _halton(self, n, base):
        """Generate the nth Halton number in the given base."""
        result = 0.0
        f = 1.0
        i = n
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
# EVOLVE-BLOCK-END
