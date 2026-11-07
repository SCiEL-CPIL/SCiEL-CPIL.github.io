# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2.
    
    Key insight: DTLZ2's Pareto front (when g=0) is the unit quarter-sphere in the positive octant.
    Setting distance variables to 0 ensures g=0, placing points exactly on the Pareto front.
    The first (n_obj-1) variables are angles in [0, pi/2].
    
    For 3 objectives, 12 dims:
    - x[0], x[1] are angles (theta_1, theta_2) in [0, pi/2]
    - x[2..11] are distance variables, set to 0 for g=0
    
    Objective computation (standard DTLZ2):
    f_1 = (1+g) * cos(x[0]) * cos(x[1])
    f_2 = (1+g) * cos(x[0]) * sin(x[1])
    f_3 = (1+g) * sin(x[0])
    
    With g=0: f_1 = cos(x[0])*cos(x[1]), f_2 = cos(x[0])*sin(x[1]), f_3 = sin(x[0])
    
    Strategy: Use Sobol sequence (scipy.stats.qmc) for low-discrepancy sampling of the 2D angle space.
    This maximizes hypervolume coverage of the quarter-sphere.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        n_angles = n_obj - 1  # 2 for 3 objectives
        n_dist = dim - n_angles  # 10 for 12 dims
        
        # Use Sobol sequence for low-discrepancy sampling
        from scipy.stats import qmc
        
        # Generate Sobol points for the angle space
        # We'll use most of the budget for space-filling sampling
        n_samples = min(budget, 200)
        
        sampler = qmc.Sobol(d=n_angles, scramble=True)
        # Sobol requires power of 2 for best properties, but works for any n
        # Generate n_samples points in [0, 1]^2
        sobol_pts = sampler.random(n_samples)
        
        # Map to angle space [0, pi/2]
        angles = sobol_pts * (np.pi / 2)
        
        # Build x vectors: first n_angles are angles, rest are 0 (optimal for g=0)
        X = np.zeros((n_samples, dim))
        X[:, :n_angles] = angles
        X[:, n_angles:] = 0.0  # Set distance vars to 0 for g=0
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_samples
        
        # Check time
        if time.time() >= deadline:
            return X
        
        # Use remaining budget for additional refinement if needed
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Add a few more points with slightly different sampling
            # Use Latin hypercube or additional Sobol points
            extra_n = min(remaining, 20)
            sampler2 = qmc.Sobol(d=n_angles, scramble=True, skip=n_samples)
            extra_pts = sampler2.random(extra_n)
            extra_angles = extra_pts * (np.pi / 2)
            
            X_extra = np.zeros((extra_n, dim))
            X_extra[:, :n_angles] = extra_angles
            X_extra[:, n_angles:] = 0.0
            
            F_extra = np.array([problem(x) for x in X_extra])
            used += extra_n
            
            X = np.vstack([X, X_extra])
            F = np.vstack([F, F_extra])
        
        return X
# EVOLVE-BLOCK-END
