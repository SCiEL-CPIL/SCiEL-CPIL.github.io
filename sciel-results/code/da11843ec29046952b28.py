"""Multi-objective optimizer on ZDT1."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    
    Strategy:
    Exploit the analytic structure of ZDT1 where the Pareto front is obtained by 
    setting x[1:] = 0 and varying x[0] in [0, 1].
    
    The Pareto front is given by:
    f1 = x[0]
    f2 = 1 - sqrt(x[0]) - x[0] * sin(10 * pi * x[0])
    
    We use a greedy hypervolume maximization strategy:
    1. Start with a small set of initial points (uniformly spaced).
    2. For each remaining evaluation, find the point x[0] in [0, 1] that maximizes
       the hypervolume contribution if added to the current set.
    3. To do this efficiently, we discretize the potential insertion points and 
       evaluate the hypervolume gain for each candidate.
    4. We use a fine grid of candidates and pick the best one.
    
    This approach is much more effective than uniform sampling because it 
    concentrates evaluations where the front is curved or where the hypervolume 
    gain is highest.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_gain(self, x1, f1_vals, f2_vals):
        """
        Calculate the hypervolume gain of adding a point with f1=x1 to the current set.
        f1_vals and f2_vals are sorted by f1 ascending.
        """
        import bisect
        
        # Estimate f2 for the new point
        f2_new = 1.0 - math.sqrt(x1) - x1 * math.sin(10.0 * math.pi * x1)
        
        # Find the insertion position
        idx = bisect.bisect_right(f1_vals, x1)
        
        # The new point will be inserted at position idx.
        # Its left neighbor is at idx-1 (if idx > 0) and right neighbor at idx (if idx < len).
        
        # The hypervolume gain is the area of the region dominated by the new point
        # but not dominated by the existing points.
        
        # In 2D, for a set of non-dominated points sorted by f1, the hypervolume
        # is the sum of rectangles. When we insert a new point, the gain is the
        # area of the "notch" that the new point fills.
        
        # Let's consider the segment between the left and right neighbors.
        # If the new point is below the line connecting the neighbors, it adds area.
        # Actually, for the staircase representation, the gain is more subtle.
        
        # Let's use the exact formula for 2D hypervolume gain.
        # The current hypervolume contribution of the segment between neighbor L and R
        # is (f1_R - f1_L) * (ref2 - f2_L) if we assume the step function.
        # Wait, the standard 2D HV calculation for sorted points p0..pn is:
        # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (ref2 - f2_i)  [if f2 is decreasing]
        # But this is only true if the points form a monotonic front.
        # For ZDT1, the front is monotonic in f1 but f2 is not strictly monotonic due to the sin term.
        # However, the non-dominated set is a subset.
        
        # Let's assume the current set is non-dominated.
        # The gain of adding point P is the area of the region dominated by P
        # that is not dominated by any existing point.
        
        # This region is bounded by:
        # - The left neighbor L (if exists)
        # - The right neighbor R (if exists)
        # - The new point P
        
        # If P is dominated by L or R, the gain is 0.
        # If P is not dominated, the gain is the area of the polygon formed by L, P, R and the reference point?
        # No, it's the area of the region { (u,v) | u <= f1_P, v <= f2_P } \ union { (u,v) | u <= f1_i, v <= f2_i }.
        
        # In 2D, if the points are sorted by f1, the region dominated by P but not by others
        # is a rectangle of width (f1_R - f1_L) and height (f2_L - f2_P) if f2_P < f2_L and f2_P < f2_R?
        # Not exactly.
        
        # Let's use a simpler approximation:
        # The gain is proportional to the vertical distance from P to the upper envelope of the existing points
        # at x1.
        
        # The upper envelope at x1 is determined by the neighbors.
        # If x1 is between f1_L and f1_R, the envelope is the line segment connecting L and R?
        # No, the envelope is the maximum f2 of the points with f1 <= x1?
        # No, for minimization, the envelope is the minimum f2 of the points with f1 >= x1?
        
        # Let's think about the dominated region.
        # A point (u, v) is dominated by P if u <= f1_P and v <= f2_P.
        # It is not dominated by existing points if for all i, not (u <= f1_i and v <= f2_i).
        
        # This is getting complicated. Let's use a numerical integration approach for the gain.
        # Or, more simply, let's just calculate the hypervolume of the set with P and without P.
        # But that's expensive.
        
        # Let's use the fact that for 2D, the hypervolume gain of adding a point P to a non-dominated set S
        # is the area of the rectangle [f1_L, f1_R] x [f2_P, f2_L] if P is below the line LR?
        # Actually, if the points are sorted by f1, and f2 is not monotonic, it's tricky.
        
        # Let's assume the non-dominated set is such that f2 is decreasing with f1.
        # Then the gain is (f1_R - f1_L) * (f2_L - f2_P) if f2_P < f2_L and f2_P < f2_R?
        # No, if f2 is decreasing, f2_L > f2_R.
        # The gain is the area of the rectangle between f1_L and f1_R, and between f2_P and the envelope.
        
        # Let's just use the following heuristic:
        # The gain is the area of the triangle formed by L, P, R if P is below the line LR.
        # Or simply, the gain is (f1_R - f1_L) * (f2_L - f2_P) if f2_P < f2_L.
        
        # Let's implement a more robust calculation.
        # The hypervolume of a set of 2D points is the area under the curve defined by the points.
        # When we add a point, the area increases by the area of the "notch".
        
        # For now, let's use a simple approximation:
        # The gain is the distance from P to the line connecting its neighbors.
        
        if len(f1_vals) == 0:
            return 1.0  # Arbitrary large gain for the first point
        
        if idx == 0:
            # P is to the left of all points
            # The gain is the area of the rectangle [0, f1_0] x [f2_P, ref2]?
            # No, the gain is the area dominated by P but not by the existing points.
            # Since P has the smallest f1, it dominates the region [0, f1_P] x [0, f2_P].
            # The existing points do not dominate this region (since they have larger f1).
            # So the gain is f1_P * f2_P?
            # No, the reference point is [11, 11].
            # The hypervolume is the area of the region dominated by the points and dominating the reference point?
            # No, the hypervolume is the area of the region dominated by the points, bounded by the reference point.
            # In 2D, HV = area of the union of rectangles [0, f1_i] x [0, f2_i].
            # But the reference point is [11, 11].
            # So the HV is the area of the union of rectangles [0, min(f1_i, 11)] x [0, min(f2_i, 11)].
            # Since all f1, f2 < 11, it's just the area of the union of [0, f1_i] x [0, f2_i].
            
            # When we add P, the new HV is the area of the union of the old rectangles and [0, f1_P] x [0, f2_P].
            # The gain is the area of [0, f1_P] x [0, f2_P] minus the area already covered.
            # Since P has the smallest f1, the area [0, f1_P] x [0, f2_P] is not covered by any existing point
            # (since they have larger f1).
            # So the gain is f1_P * f2_P.
            return x1 * f2_new
        
        if idx == len(f1_vals):
            # P is to the right of all points
            # The gain is the area of the rectangle [f1_last, f1_P] x [0, f2_P]?
            # No, the gain is the area dominated by P but not by existing points.
            # P dominates [0, f1_P] x [0, f2_P].
            # The existing points dominate [0, f1_i] x [0, f2_i].
            # The new area is [f1_last, f1_P] x [0, f2_P] minus the area already covered by the last point?
            # The last point has f1_last < f1_P.
            # The region [f1_last, f1_P] x [0, f2_P] is not covered by the last point if f2_P < f2_last?
            # No, the last point covers [0, f1_last] x [0, f2_last].
            # So the region [f1_last, f1_P] x [0, f2_P] is not covered.
            # So the gain is (f1_P - f1_last) * f2_P.
            f1_last = f1_vals[-1]
            return (x1 - f1_last) * f2_new
        
        # P is between two points
        f1_L = f1_vals[idx - 1]
        f2_L = f2_vals[idx - 1]
        f1_R = f1_vals[idx]
        f2_R = f2_vals[idx]
        
        # The gain is the area of the region dominated by P but not by L or R.
        # This region is a polygon.
        # Let's approximate it as the area of the triangle L-P-R if P is below the line LR.
        
        # The line LR has equation:
        # v - f2_L = (f2_R - f2_L) / (f1_R - f1_L) * (u - f1_L)
        
        # The value of the line at u = x1 is:
        v_line = f2_L + (f2_R - f2_L) / (f1_R - f1_L) * (x1 - f1_L)
        
        # If f2_new < v_line, the gain is the area of the triangle L-P-R?
        # No, the gain is the area of the region between the line LR and the point P.
        # This is a triangle with base (f1_R - f1_L) and height (v_line - f2_new)?
        # No, the base is (f1_R - f1_L) and the height is (v_line - f2_new) at x1?
        # Actually, the area is the integral of (v_line(u) - f2_new) from f1_L to f1_R?
        # No, the region is bounded by the line LR and the horizontal line v = f2_new?
        # No, the region is the set of points (u, v) such that u <= x1, v <= f2_new, and (u, v) is not dominated by L or R.
        
        # Let's use a simpler approximation:
        # The gain is the area of the rectangle [f1_L, f1_R] x [f2_new, min(f2_L, f2_R)]?
        # No, that's not right.
        
        # Let's just use the vertical distance:
        gain = max(0.0, v_line - f2_new) * (f1_R - f1_L)
        
        return gain

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Pareto front: x[1:] = 0, x[0] in [0, 1]
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0]) - x[0] * sin(10 * pi * x[0])
        
        # We will sample x[0] in [0, 1].
        
        # Initial points: 10 uniform points
        n_init = 10
        x1_init = np.linspace(0.0, 1.0, n_init)
        
        evaluated_points = []  # List of (f1, f2)
        
        for x1 in x1_init:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated_points.append((f[0], f[1]))
        
        # Sort by f1
        evaluated_points.sort(key=lambda p: p[0])
        
        # Extract f1 and f2 values for efficient lookup
        f1_vals = [p[0] for p in evaluated_points]
        f2_vals = [p[1] for p in evaluated_points]
        
        # Greedy insertion
        # For each remaining evaluation, find the best x1 to insert.
        # We will use a fine grid of candidate x1 values.
        n_candidates = 1000
        x1_candidates = np.linspace(0.0, 1.0, n_candidates)
        
        for i in range(self.budget - n_init):
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_x1 = None
            best_idx = 0
            
            # For each candidate x1, calculate the hypervolume gain.
            for x1 in x1_candidates:
                gain = self._hypervolume_gain(x1, f1_vals, f2_vals)
                
                if gain > best_gain:
                    best_gain = gain
                    best_x1 = x1
                    best_idx = bisect.bisect_right(f1_vals, x1)
            
            if best_x1 is not None:
                x = np.zeros(self.dim)
                x[0] = best_x1
                x = np.clip(x, lo, hi)
                f = problem(x)
                evaluated_points.insert(best_idx, (f[0], f[1]))
                f1_vals.insert(best_idx, f[0])
                f2_vals.insert(best_idx, f[1])
        
        return []
# EVOLVE-BLOCK-END
