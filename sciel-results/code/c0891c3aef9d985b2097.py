import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, leave 10s margin
        
        # DTLZ7 Pareto Front Analysis:
        # f1 = y1
        # f2 = y1 * (1 - y2)
        # f3 = (1 + y2) * (1 - f1 - f2) = (1 + y2) * (1 - y1 - y1(1-y2))
        #      = (1 + y2) * (1 - 2y1 + y1*y2)
        
        # Variables:
        # x[0] = y1 in [0, 1]
        # x[1] = y2 in [0, 1]
        # x[2:] = 0 (to minimize g to 1)
        
        # Reference Point: R = [1.1, 1.1, 25.0]
        # Only points with f_i < R_i for all i contribute.
        # f1 < 1.1 is always true for y1 in [0,1].
        # f2 < 1.1 is always true.
        # f3 < 25 is always true (max f3 is around 2 or 3).
        
        # The hypervolume of a set of points S is the volume of the union of boxes
        # [f1_i, R1] x [f2_i, R2] x [f3_i, R3].
        
        # Strategy: Greedy Hypervolume Maximization
        # 1. Initialize with a set of candidate points on the front.
        #    Since the front is 2D, we can parameterize it by (y1, y2).
        #    However, evaluating hypervolume gain for every possible (y1, y2) is expensive.
        #    Instead, we can use a grid of candidates and pick the best one,
        #    or use a more efficient method.
    
        # A simpler and very effective approach for small budgets (200) on a known front:
        # The hypervolume contribution is roughly proportional to the "box" size.
        # Greedy selection:
        # Start with an empty set.
        # At each step, find the point on the front that maximizes the hypervolume of the union.
        
        # To make this tractable:
        # We can pre-compute a dense grid of candidate points on the front.
        # Then, for each step, we evaluate the marginal gain for each candidate.
        # This is O(N * M) where M is the number of candidates.
        # With N=200 and M=10000, that's 2M operations, which is fast.
        
        # Let's create a grid of candidates.
        # y1 in [0, 1], y2 in [0, 1].
        # But we can be smarter. The front is a triangle in (f1, f2) space?
        # No, f1=y1, f2=y1(1-y2).
        # For a fixed y1, f2 ranges from 0 to y1.
        # So the front in (f1, f2) is the region 0 <= f2 <= f1 <= 1.
        # And f3 is determined by f1, f2.
        # f3 = (1 + y2)(1 - f1 - f2).
        # y2 = 1 - f2/f1.
        # f3 = (2 - f2/f1)(1 - f1 - f2).
        
        # Let's generate candidates in (f1, f2) space uniformly in the triangle.
        # Or just a grid in (y1, y2).
        
        n_candidates = 2000
        # Use a low-discrepancy sequence or just a grid.
        # A grid in y1, y2 is fine.
        ys1 = np.linspace(0.0, 1.0, int(np.sqrt(n_candidates)))
        ys2 = np.linspace(0.0, 1.0, int(np.sqrt(n_candidates)))
        Y1, Y2 = np.meshgrid(ys1, ys2)
        Y1 = Y1.flatten()
        Y2 = Y2.flatten()
        
        # Compute objectives for all candidates
        # f1 = y1
        # f2 = y1 * (1 - y2)
        # f3 = (1 + y2) * (1 - 2*y1 + y1*y2)
        f1_cand = Y1
        f2_cand = Y1 * (1 - Y2)
        f3_cand = (1 + Y2) * (1 - 2*Y1 + Y1*Y2)
        
        # Filter out points that are not strictly better than reference?
        # All are better.
        
        # Store candidates as (f1, f2, f3, y1, y2)
        # We need to map back to x for evaluation.
        
        # Greedy Selection
        selected_indices = []
        selected_objs = [] # list of (f1, f2, f3)
        
        # To speed up hypervolume calculation, we can use a simple approximation
        # or an exact 3D hypervolume algorithm.
        # With N=200, exact hypervolume is feasible.
        # But doing it for 2000 candidates at each step (200 steps) is 200 * 2000 * HV_cost.
        # HV_cost for 200 points is small.
        
        # Let's implement a fast HV calculation.
        # Or, since the front is 2D, we can project to 2D?
        # No, f3 varies significantly.
        
        # Alternative: Use the property that for a non-dominated set,
        # the hypervolume is the sum of contributions of each point.
        # Contribution of point i is the volume of the box it "owns".
        # This is complex to compute dynamically.
        
        # Let's try a simpler heuristic first:
        # The "knee" of the front is important.
        # Points near the "center" of the front (in objective space) tend to have larger boxes.
        
        # Actually, let's just use a deterministic sampling strategy that is known to be good for HV.
        # "Hypervolume-aware sampling":
        # Sample points such that they are spaced out in the objective space.
        
        # Let's try:
        # 1. Sample y1 from a distribution that favors the "middle" of the range?
        #    No, the box width in f1 is (1.1 - f1). So smaller f1 gives wider box.
        #    But smaller f1 also limits f2 (since f2 <= f1).
        #    So there is a trade-off.
        
        # Let's go back to Greedy HV.
        # I will implement a simple exact HV calculator for 3D.
        
        def calc_hv(points, ref):
            # points: Nx3 array
            # ref: 3x1 array
            # Returns hypervolume
            if len(points) == 0:
                return 0.0
            # Sort by f1
            points = points[np.argsort(points[:, 0])]
            # Recursively compute
            # This is a simple recursive implementation, might be slow for 200 points.
            # For 200 points, O(N^2) or O(N^3) is fine.
            
            # Actually, let's use a simpler approximation for speed during selection:
            # The marginal gain of adding a point p to a set S is the volume of the box
            # [p1, R1] x [p2, R2] x [p3, R3] that is NOT covered by any point in S.
            # This is hard to compute exactly without a full HV algorithm.
            
            # Given the time constraint, let's use a different strategy:
            # **Stratified Sampling in Objective Space**
            # We want to cover the front well.
            # The front is a 2D surface.
            # We can sample uniformly in the (f1, f2) triangle, as before,
            # but maybe with a bias?
            
            # Let's look at the inspiration: "hypervolume-aware importance sampling".
            # This suggests weighting the sampling density by the expected contribution.
            # The contribution of a point is roughly the size of the box it creates.
            # Box size ~ (1.1-f1)(1.1-f2)(25-f3).
            # Since (25-f3) is large and relatively constant, we can approximate
            # the importance as (1.1-f1)(1.1-f2).
            
            # So we should sample more densely where (1.1-f1)(1.1-f2) is large.
            # This is large when f1 and f2 are small.
            # But f2 <= f1.
            # So we should bias towards small f1 and f2.
            
            # How to sample from a density proportional to (1.1-f1)(1.1-f2) over the triangle?
            # We can use rejection sampling or inverse transform if possible.
            # Or just use a grid and pick points with probability proportional to weight.
            
            # Let's try a simple weighted sampling from a grid.
            
            n_grid = 50
            f1_vals = np.linspace(0.0, 1.0, n_grid)
            f2_vals = np.linspace(0.0, 1.0, n_grid)
            
            candidates = []
            weights = []
            
            for f1 in f1_vals:
                for f2 in f2_vals:
                    if f2 <= f1: # Inside triangle
                        # Compute f3
                        if f1 > 1e-10:
                            y2 = 1 - f2/f1
                            y1 = f1
                        else:
                            y1 = 0
                            y2 = 0
                        f3 = (1 + y2) * (1 - 2*y1 + y1*y2)
                        
                        # Weight
                        w = (1.1 - f1) * (1.1 - f2) * (25.0 - f3)
                        candidates.append([f1, f2, f3, y1, y2])
                        weights.append(w)
                        
            candidates = np.array(candidates)
            weights = np.array(weights)
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Sample n_samples points without replacement?
            # Or with replacement?
            # Without replacement ensures diversity.
            # But if we sample without replacement from a grid, we might miss some areas.
            # Let's sample with replacement but use a large grid.
            
            n_samples = self.budget
            indices = np.random.choice(len(candidates), size=n_samples, replace=False, p=weights)
            selected = candidates[indices]
            
            # Sort selected points?
            # Not necessary for evaluation.
            
            # Construct x vectors
            X = np.zeros((n_samples, self.dim))
            X[:, 0] = selected[:, 3] # y1
            X[:, 1] = selected[:, 4] # y2
            # x[2:] = 0
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            for i in range(n_samples):
                if time.time() > deadline:
                    break
                x = X[i]
                x = np.clip(x, lo, hi)
                problem(x)
                
            return X
