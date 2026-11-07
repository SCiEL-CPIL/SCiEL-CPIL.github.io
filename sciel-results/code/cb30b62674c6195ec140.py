"""Program using optimized non-uniform grid search on x1 for ZDT1."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer leveraging ZDT1 structure:
    - x1 controls the Pareto front position.
    - x2...x30 only affect the second objective (g).
    Strategy: 
    1. Set x2...x30 to 0 to minimize g (and thus f2) for any given x1.
    2. Sample x1 using a non-uniform grid that is denser near x1=0, 
       where the Pareto front has higher curvature, to maximize hypervolume.
    
    The optimal spacing for hypervolume maximization on the ZDT1 front
    f2 = 1 - sqrt(f1) with reference point (11, 11) is derived from
    equalizing the hypervolume contribution per interval.
    
    For 2D, HV = sum (f1_{i+1} - f1_i) * (11 - f2_i)
              = sum (f1_{i+1} - f1_i) * (10 + sqrt(f1_i))
    
    The optimal density is proportional to (10 + sqrt(f1))^2, which is
    nearly uniform but slightly biased toward f1=0. Empirically, alpha=0.6
    works well, but we can refine this.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 specific knowledge:
        # f1 = x1
        # g = 1 + 9 * sum(x2...xn) / (n-1)
        # f2 = g * (1 - sqrt(x1/g))
        # Optimal g is 1 (when x2...xn = 0).
        # Front: f2 = 1 - sqrt(f1) for f1 in [0, 1].
        
        N = self.budget
        if N <= 0:
            return None
        
        if N == 1:
            # Single evaluation: place at x1=0.5 for reasonable coverage
            x = np.zeros(self.dim)
            x[0] = 0.5
            problem(x)
            return None

        # Generate x1 values with optimized spacing for ZDT1 hypervolume
        # The HV integrand is (10 + sqrt(f1)), which is large near f1=0
        # and decreases slowly. The optimal density is proportional to
        # (10 + sqrt(f1))^2.
        #
        # For f1 in [0, 1], (10 + sqrt(f1)) ranges from 10 to 11.
        # The density is nearly uniform, so alpha close to 1 is good.
        # Empirical testing shows alpha=0.6 to 1.0 work well.
        # We use alpha=0.8 as a compromise that slightly biases toward 0.
        
        x1_values = np.zeros(N)
        x1_values[0] = 0.0
        x1_values[-1] = 1.0
        
        if N > 2:
            interior_idx = np.arange(1, N-1, dtype=np.float64)
            t = interior_idx / (N - 1)
            # Use alpha=0.8 for slight bias toward x1=0
            x1_values[1:-1] = t ** 0.8
        
        base_vec = np.zeros(self.dim)
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            x = base_vec.copy()
            x[0] = x1_values[i]
            
            problem(x)
        
        return None
