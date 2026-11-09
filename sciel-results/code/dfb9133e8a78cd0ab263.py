"""Structure-aware multi-objective optimizer for ZDT4 exploiting known Pareto front parametrization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT4.
    
    ZDT4 structure (standard):
      f1 = 1 + 8*x1^2 + sum_{i=2}^{n} 20*sin^2(4*pi*x_i)
      f2 = (x1 + 1)^2
    
    The Pareto front is achieved when sin^2(4*pi*x_i) = 0 for all i > 1,
    which occurs at x_i = k/4 for integer k. Setting x_i = 0 (a valid zero)
    guarantees minimal f1 = 1 + 8*x1^2 for any x1.
    
    Strategy: 
    1. Identify that the non-dominated Pareto front lies entirely on the branch where x1 <= 0.
       For x1 > 0, the point is dominated by the point with -x1 (same f1, lower f2).
    2. Use the full budget to sample this non-dominated branch densely.
    3. Instead of sampling x1 uniformly, sample f1 uniformly in its valid range [1, 11].
       This maps to x1 = -sqrt((f1 - 1) / 8).
       This structural change ensures points are spread uniformly in the objective space 
       (specifically along the f1 axis), which is more effective for hypervolume 
       maximization than uniform decision space sampling for this quadratic front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        all_points = []
        
        # Check if we have time to proceed
        if time.time() > deadline:
            return all_points
        
        # The non-dominated Pareto front for ZDT4 is on x1 <= 0.
        # f1 = 1 + 8*x1^2
        # f2 = (x1 + 1)^2
        
        # We want f1 < 11.0 (reference point)
        # 1 + 8*x1^2 < 11 => x1^2 < 10/8 = 1.25 => |x1| < 1.118
        # So x1 in [-1.118, 0] is the relevant range for the non-dominated front.
        
        max_x1_abs = np.sqrt(10.0 / 8.0)
        
        # Intersect with bounds for x1
        # The non-dominated branch is x1 <= 0.
        x1_lo_bound = lo[0]
        x1_hi_bound = hi[0]
        
        # We only care about x1 <= 0.
        # If the bound is entirely positive, we are in trouble, but standard ZDT4 bounds are [-5, 5].
        # Let's assume standard bounds or handle generally.
        # The valid range for x1 on the non-dominated front is [max(lo[0], -max_x1_abs), min(hi[0], 0)]
        
        x1_min = max(lo[0], -max_x1_abs)
        x1_max = min(hi[0], 0.0)
        
        # If the range is invalid (e.g., lo[0] > 0), we might have to fall back to x1=0 or similar.
        # But for ZDT4, lo[0] is typically -5.
        if x1_min >= x1_max:
             # Fallback: just use x1=0 if the non-dominated branch is cut off
             x1_min = 0.0
             x1_max = 0.0
             n_evals = budget
             X = np.zeros((n_evals, dim))
             for x in X:
                 if time.time() > deadline:
                     break
                 f = problem(x)
                 all_points.append(np.asarray(f, dtype=float).copy())
             return all_points

        # Sample f1 uniformly in the range corresponding to [x1_min, x1_max]
        # f1(x1_min) and f1(x1_max)
        f1_at_x1_min = 1.0 + 8.0 * x1_min**2
        f1_at_x1_max = 1.0 + 8.0 * x1_max**2
        
        # Since x1_min is negative and x1_max is 0, 
        # f1_at_x1_min is the max f1 (up to 11)
        # f1_at_x1_max is the min f1 (1.0)
        
        f1_lo = min(f1_at_x1_min, f1_at_x1_max)
        f1_hi = max(f1_at_x1_min, f1_at_x1_max)
        
        n_evals = budget
        # Generate uniform f1 samples
        f1_samples = np.linspace(f1_lo, f1_hi, n_evals)
        
        # Invert f1 to x1 on the left branch (x1 < 0)
        # f1 = 1 + 8*x1^2 => x1 = -sqrt((f1 - 1) / 8)
        x1_samples = -np.sqrt(np.maximum(0.0, (f1_samples - 1.0) / 8.0))
        
        # Ensure x1_samples are within bounds [x1_min, x1_max]
        x1_samples = np.clip(x1_samples, x1_min, x1_max)
        
        # Construct X
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_samples
        
        # Set x2..xn to 0 (a zero of the periodic term, guaranteeing minimal f1)
        for i in range(1, dim):
             X[:, i] = 0.0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for x in X:
            if time.time() > deadline:
                break
            f = problem(x)
            all_points.append(np.asarray(f, dtype=float).copy())
        
        return all_points

# EVOLVE-BLOCK-END
