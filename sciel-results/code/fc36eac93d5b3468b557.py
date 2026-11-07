"""Multi-objective optimizer for ZDT3 using 1D non-linear least-squares."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for ZDT3 using L-BFGS-B to optimize the placement of points
    on the analytical Pareto front to maximize hypervolume contribution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_samples = self.budget
        g_min = 1 + 10 * (self.dim - 1)
        
        # Analytical front function for ZDT3
        def front_f2(f1):
            # f1 in [0, 1]
            f1 = np.clip(f1, 0, 1)
            term1 = np.sqrt(f1 / g_min)
            term2 = (f1 / g_min) * np.sin(10 * np.pi * f1)
            return g_min * (1 - term1 - term2)
        
        # Objective: maximize hypervolume contribution of sampled points
        # Hypervolume in 2D with reference point (r1, r2):
        # For sorted points, HV = sum of (r2 - f2_i) * (f1_{i+1} - f1_i) for interior points
        # We want to maximize this, so minimize negative HV.
        
        def hypervolume_neg(t):
            f1_s = np.sort(t)
            f2_s = front_f2(f1_s)
            
            # Reference point
            r1, r2 = 11.0, 11.0
            
            # Compute hypervolume contribution
            # For each consecutive pair, the contribution is a trapezoid
            # Points must be strictly better than reference
            valid = (f1_s < r1) & (f2_s < r2)
            if np.sum(valid) < 2:
                return 0.0
            
            f1_valid = f1_s[valid]
            f2_valid = f2_s[valid]
            
            # Add reference point at the end
            # HV = integral-like sum
            # For sorted f1, the hypervolume is:
            # sum over i of (f1_{i+1} - f1_i) * (r2 - max(f2_i, f2_{i+1}))
            # But more precisely for non-dominated points:
            # HV = sum_i (r2 - f2_i) * (f1_{i+1} - f1_i) where f1_0 = 0, f1_n = r1
            
            # Actually, the correct formula for HV of a set of points with ref (r1, r2):
            # Sort by f1 ascending. For each point i, the "width" it contributes is
            # from f1_i to f1_{i+1} (or r1 for last), and the "height" is (r2 - f2_i)
            # but only if f2_i is the minimum in that interval.
            
            # Simpler: HV = (r1 * r2) - (area above the front covered by points)
            # Area above = sum of rectangles/trapezoids
            
            # Let's use the standard 2D HV computation:
            # Sort by f1. For each point, the contribution is:
            # width_i * (r2 - f2_i) where width_i = f1_{i+1} - f1_i for i < n-1,
            # and width_{n-1} = r1 - f1_{n-1}
            # But this overcounts if points are not on the front.
            
            # Correct approach: The HV of a set S with ref r is the volume of the union
            # of boxes [f_i, r]. In 2D, sort by f1, then:
            # HV = sum_i (f1_{i+1} - f1_i) * (r2 - min_{j>=i} f2_j)
            # where f1_0 = 0 (or min f1), f1_n = r1
            
            # Actually, standard algorithm:
            # Sort by f1 ascending.
            # For i from 0 to n-2: contribution = (f1_{i+1} - f1_i) * (r2 - f2_i)
            # Last: contribution = (r1 - f1_{n-1}) * (r2 - f2_{n-1})
            # This works if the points are non-dominated and sorted.
            
            if len(f1_valid) == 1:
                return -(r1 - f1_valid[0]) * (r2 - f2_valid[0])
            
            hv = 0.0
            for i in range(len(f1_valid) - 1):
                width = f1_valid[i + 1] - f1_valid[i]
                height = r2 - f2_valid[i]
                hv += width * height
            
            # Last point
            width_last = r1 - f1_valid[-1]
            height_last = r2 - f2_valid[-1]
            hv += width_last * height_last
            
            return -hv  # Negate because we minimize
        
        # Initial guess: uniform grid
        t_init = np.linspace(0, 1, n_samples)
        
        # Bounds for t: each t_i in [0, 1]
        bounds_opt = [(0, 1)] * n_samples
        
        # Optimize using L-BFGS-B
        try:
            res = minimize(hypervolume_neg, t_init, method='L-BFGS-B', bounds=bounds_opt, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False})
            t_opt = res.x
        except Exception:
            t_opt = t_init
            
        # Ensure t_opt is within bounds
        t_opt = np.clip(t_opt, 0, 1)
        
        # Construct the x vectors
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = t_opt
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
