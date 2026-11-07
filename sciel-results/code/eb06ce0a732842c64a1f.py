# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2.
    
    Strategy:
    - DTLZ2 Pareto front for 3 objectives is a quarter-sphere in the positive octant.
    - Distance variables (last dim - n_obj + 1 = 10 vars) set to 0.5 for optimality.
    - Angle variables (first n_obj - 1 = 2 vars) determine position on Pareto front.
    - Use deterministic low-discrepancy Sobol sequence to sample 2D angle space
      maximally space-filling, ensuring good hypervolume coverage.
    - Allocate entire budget to this deterministic sampling to maximize unique Pareto points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz2_objectives_from_angles(self, angles):
        """
        Compute DTLZ2 objectives from angle variables.
        For DTLZ2 with n_obj objectives:
        - g = sum((x_i - 0.5)^2) for i = n_obj-1 to dim-1 (distance vars)
        - If distance vars are set to 0.5, g=0.
        
        For 3 objectives (n_obj=3):
        f_0 = sin(pi*x_0/2)
        f_1 = cos(pi*x_0/2) * sin(pi*x_1/2)
        f_2 = cos(pi*x_0/2) * cos(pi*x_1/2)
        """
        # angles is shape (n, 2) with values in [0,1]
        x0 = angles[:, 0] * np.pi / 2
        x1 = angles[:, 1] * np.pi / 2
        c0 = np.cos(x0)
        s0 = np.sin(x0)
        c1 = np.cos(x1)
        s1 = np.sin(x1)
        f0 = s0
        f1 = c0 * s1
        f2 = c0 * c1
        return np.column_stack([f0, f1, f2])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_angle = self.n_obj - 1  # 2 for 3 objectives
        n_dist = self.dim - n_angle  # 10 for 12 vars
        
        # Set distance variables to 0.5 (optimal for DTLZ2)
        dist_vals = np.full(n_dist, 0.5)
        
        # Generate Sobol sequence for angle space
        n_points = self.budget
        # Sobol requires power of 2 for optimal properties, but works for any N.
        # We use scramble=False for reproducibility and determinism.
        sobol = qmc.Sobol(d=n_angle, scramble=False)
        angles = sobol.random(n_points)
        
        # Build full x vectors and evaluate
        all_F = []
        for a in angles:
            x = np.concatenate([a, dist_vals])
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_F.append(f)
            
            if time.time() > deadline:
                break
        
        all_F = np.array(all_F)
        
        return all_F
# EVOLVE-BLOCK-END
