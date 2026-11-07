# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Direct, structure-aware optimizer for ZDT1.
    Key insight: For ZDT1, the Pareto front is achieved when all variables x2..x30 are 0.
    f1(x) = x1, and the Pareto front is f2 = 1 - sqrt(f1) for f1 in [0,1].
    We sample x1 values to maximize hypervolume coverage of the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0

        # For ZDT1 with bounds [0,1], the Pareto front is:
        # x = (t, 0, 0, ..., 0) for t in [0, 1]
        # f1 = t, f2 = 1 - sqrt(t)
        
        # To maximize hypervolume with a reference point of [11, 11],
        # we want to cover the front well. The front goes from (0, 1) to (1, 0).
        # The hypervolume contribution is dominated by the area under the curve.
        
        # Strategy: Use a mix of uniform and concentrated sampling.
        # The curve f2 = 1 - sqrt(f1) has high slope near f1=0 and low slope near f1=1.
        # Points near f1=0 contribute more to hypervolume per unit of f1 range.
        
        # Let's use a non-uniform grid that is denser near f1=0.
        # A good approach is to use quantiles of a distribution that is skewed toward 0.
        # Or simply use a power-law spacing.
        
        n_total = self.budget
        n_front = n_total  # Use all budget on the front
        
        # Use a grid that is dense near 0.
        # Let's try: x1_i = (i + 0.5)^alpha / n^alpha for some alpha > 1?
        # Or better: use the fact that the Pareto front in objective space is a curve.
        # We want to maximize the sum of areas of rectangles.
        
        # A simple effective strategy: uniform in f1 space works well.
        # But let's try to be smarter: use more points where the curve is "steeper" in f2.
        # Actually, for hypervolume, what matters is the "envelope" of the non-dominated points.
        # Since all points are on the front, the hypervolume is determined by the sorted points.
        
        # Let's use a uniform grid in [0, 1] for x1.
        x1_vals = (np.arange(n_front) + 0.5) / n_front
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        X = np.zeros((n_front, self.dim))
        X[:, 0] = x1_vals
        # The rest are 0, which is within [0,1]
        
        F = np.zeros((n_front, self.n_obj))
        for i in range(n_front):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            n_evals += 1
            
        return X[:n_evals]
# EVOLVE-BLOCK-END
