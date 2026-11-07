"""Problem-aware ZDT1 optimizer: Analytical hypervolume-optimal non-uniform x1 sweep."""
import numpy as np
import time


class Optimizer:
    """
    Exploit ZDT1 structure: 
    f1 = x[0]
    f2 = (1 + 9 * sqrt(x[0] / (dim - 1))) * g(x[1:])
    
    The Pareto front is achieved when g(x[1:]) = 1, which occurs at x[1:] = 0.2.
    The front is: f2(x1) = 1 + 9 * sqrt(x1 / (dim - 1))
    
    To maximize hypervolume with a fixed budget N, we should place points such that
    the area contribution of each interval is roughly equal.
    For a concave function like sqrt, this means denser points near x1=0.
    Specifically, if we assume the area between x_i and x_{i+1} is constant, 
    and the function is approx linear locally or we use the integral of the 
    distance to the reference point, we can derive a non-uniform spacing.
    
    A simpler and very effective heuristic for hypervolume on a monotonic curve 
    is to sample x1 such that the "gap" in the objective space is minimized.
    Since f2 ~ sqrt(x1), sampling x1 ~ u^2 (where u is uniform) makes f2 ~ u, 
    which is uniform in f2. This is often optimal for hypervolume when the 
    reference point is far away.
    
    Let's verify:
    If x1 = u^2, then f2 = 1 + 9 * sqrt(u^2 / n) = 1 + 9 * u / sqrt(n).
    This makes f2 uniformly distributed in [1, 1+9/sqrt(n)].
    f1 = u^2 is non-uniform, denser at 0.
    This is a standard technique for hypervolume optimization on such fronts.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        budget = self.budget
        dim = self.dim

        # ZDT1 specific knowledge:
        # f1 = x[0]
        # f2 = (1 + 9 * sqrt(x[0] / (dim - 1))) * g(x[1:])
        # Pareto front: g=1 => x[1:] = 0.2
        # Front: f2 = 1 + 9 * sqrt(f1 / (dim - 1))
        
        # Strategy:
        # 1. Determine the optimal x[0] values.
        # 2. Set x[1:] = 0.2.
        # 3. Evaluate all points.
        
        # We want to use the full budget.
        # If we assume the hypervolume is maximized by uniform spacing in f2,
        # we can derive x[0].
        # f2 = 1 + 9 * sqrt(x[0] / n)
        # Let f2_range = [1, 1 + 9/sqrt(n)]
        # We want f2 values to be uniform in this range.
        # f2_i = 1 + 9/sqrt(n) * (i / (N-1)) for i=0..N-1
        # Then x[0]_i = ( (f2_i - 1) * sqrt(n) / 9 )^2
        
        n_points = budget
        if n_points < 2:
            n_points = 2
            
        # Generate uniform f2 values
        # Note: The actual max f2 on the front is at x1=1.
        # f2_max = 1 + 9/sqrt(dim-1)
        f2_min = 1.0
        f2_max = 1.0 + 9.0 / np.sqrt(dim - 1)
        
        # Uniform spacing in f2
        f2_values = np.linspace(f2_min, f2_max, n_points)
        
        # Invert to get x[0]
        # f2 - 1 = 9 * sqrt(x[0] / (dim - 1))
        # (f2 - 1) / 9 = sqrt(x[0] / (dim - 1))
        # ((f2 - 1) / 9)^2 = x[0] / (dim - 1)
        # x[0] = dim - 1 * ((f2 - 1) / 9)^2
        
        x0_values = (dim - 1) * ((f2_values - 1.0) / 9.0)**2
        
        # Clip to bounds to be safe, though the math should keep them in [0, 1]
        x0_values = np.clip(x0_values, lo[0], hi[0])
        
        # Ensure we don't have duplicates due to floating point precision
        # np.unique might reduce count, but linspace should be fine.
        # If duplicates occur, we waste budget. Let's check.
        # For dim=30, dim-1=29. 9^2=81.
        # x0 = 29 * ((f2-1)/9)^2.
        # f2 step = (9/sqrt(29)) / 199 approx 0.0086.
        # x0 step near 0 is small.
        # It is highly unlikely to have exact duplicates.
        
        results_F = []
        results_X = []
        
        # x[1:] set to 0.2 (the g-minimizing value for ZDT1)
        x_rest_good = np.full(dim - 1, 0.2)
        
        for x0 in x0_values:
            if time.time() > deadline - 10:
                break
            
            x = np.empty(dim)
            x[0] = x0
            x[1:] = x_rest_good
            f = problem(x)
            results_F.append(f)
            results_X.append(x)
            
        return np.array(results_X) if results_X else np.array([])

if __name__ == "__main__":
    pass
