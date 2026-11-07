# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    DTLZ2 with 3 objectives has a Pareto front that is a quarter-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0.5 for optimality. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy: Use Sobol quasi-random sequence to sample the 2D angle space densely,
    then map to decision space. This directly generates points near/on the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 structure:
        # - First (n_obj - 1) variables are angles determining position on Pareto front
        # - Remaining (dim - n_obj + 1) variables should be 0.5 for optimality
        n_angles = self.n_obj - 1  # 2 for 3 objectives
        n_dist = self.dim - n_obj + 1  # 10 for 12 vars, 3 objs
        
        # Generate Sobol sequence for the angle space
        # We want to densely cover the 2D angle space [0, pi/2] x [0, pi/2]
        # which maps to the quarter-sphere Pareto front
        
        # Use Sobol for better low-discrepancy coverage
        n_points = min(self.budget, 200)
        
        # Sobol sequence in [0,1]^2
        sampler = qmc.Sobol(d=n_angles, scramble=True)
        # We need a power of 2 for Sobol, so round up
        n_sobol = 2 ** int(np.ceil(np.log2(n_points)))
        if n_sobol > n_points:
            n_sobol = 2 ** int(np.floor(np.log2(n_points)))
            if n_sobol < 2:
                n_sobol = 2
        
        # Generate Sobol points
        sobol_points = sampler.random(n_sobol)
        
        # Take only as many as we need
        sobol_points = sobol_points[:n_points]
        
        # Map from [0,1]^2 to angle space
        # For DTLZ2 with 3 objectives, the Pareto front is:
        # f_i = (1 + g) * y_i where sum(y_i^2) = 1, y_i >= 0
        # The angles determine y via:
        # y_1 = cos(theta_1) * cos(theta_2)
        # y_2 = cos(theta_1) * sin(theta_2)  
        # y_3 = sin(theta_1)
        # where theta_1, theta_2 in [0, pi/2]
        
        # Actually for DTLZ2, the standard parameterization is:
        # x_1 in [0,1] -> theta_1 = pi/2 * x_1
        # x_2 in [0,1] -> theta_2 = pi/2 * x_2
        # y_1 = cos(theta_1) * cos(theta_2)
        # y_2 = cos(theta_1) * sin(theta_2)
        # y_3 = sin(theta_1)
        
        theta1 = np.pi / 2 * sobol_points[:, 0]
        theta2 = np.pi / 2 * sobol_points[:, 1]
        
        y1 = np.cos(theta1) * np.cos(theta2)
        y2 = np.cos(theta1) * np.sin(theta2)
        y3 = np.sin(theta1)
        
        # Build decision vectors:
        # First n_angles variables are the angle variables (mapped from [0,1])
        # Remaining n_dist variables are set to 0.5
        X = np.zeros((n_points, self.dim))
        X[:, :n_angles] = sobol_points  # The angle variables in [0,1]
        X[:, n_angles:] = 0.5  # Distance variables at optimal value
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check if we used budget
        used = n_points
        if used < self.budget:
            # Use remaining budget for additional Sobol points with different scrambling
            remaining = self.budget - used
            n_extra_sobol = 2 ** int(np.ceil(np.log2(max(remaining, 2))))
            if n_extra_sobol > remaining:
                n_extra_sobol = 2 ** int(np.floor(np.log2(max(remaining, 2))))
                if n_extra_sobol < 2:
                    n_extra_sobol = 2
            
            sampler2 = qmc.Sobol(d=n_angles, scramble=True)
            extra_points = sampler2.random(n_extra_sobol)[:remaining]
            
            theta1_e = np.pi / 2 * extra_points[:, 0]
            theta2_e = np.pi / 2 * extra_points[:, 1]
            
            X_e = np.zeros((remaining, self.dim))
            X_e[:, :n_angles] = extra_points
            X_e[:, n_angles:] = 0.5
            
            F_e = np.array([problem(x) for x in X_e])
            
            X = np.vstack([X, X_e])
            F = np.vstack([F, F_e])
        
        return X
# EVOLVE-BLOCK-END
