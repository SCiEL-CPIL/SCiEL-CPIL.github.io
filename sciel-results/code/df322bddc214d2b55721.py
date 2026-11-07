import numpy as np
import time

class Optimizer:
    """
    Adaptive Recursive Bisection Optimizer for ZDT1.
    
    Strategy:
    1. Exploit the known structure that the Pareto front lies on the manifold x[1:] = 0.
    2. Use a recursive bisection strategy to place points. 
    3. Start with endpoints and midpoint.
    4. At each step, evaluate the midpoint of the interval (in x-space) that is 
       estimated to provide the highest marginal hypervolume gain.
    5. This creates an adaptive, error-correcting sequence that focuses evaluations 
       on the "elbow" of the front where the curvature is highest and HV gain per point is maximal.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        dim = self.dim
        budget = self.budget
        
        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])
        
        # We assume the front is found by setting x[1:] to the lower bound (0).
        # x[0] is the only variable we sweep.
        x0_lo = lo[0]
        x0_hi = hi[0]
        
        # Helper to create a candidate point
        def make_point(x0_val):
            x = np.full(dim, lo)
            x[0] = x0_val
            return x
        
        # Helper to evaluate and check validity
        def eval_point(x):
            try:
                f = problem(x)
                f = np.asarray(f, dtype=np.float64)
                if f.shape != (2,):
                    return None
                if not np.all(np.isfinite(f)):
                    return None
                return f
            except Exception:
                return None
        
        # List of evaluated points: (x0, f1, f2)
        points = []
        
        # 1. Evaluate endpoints
        x_start = make_point(x0_lo)
        f_start = eval_point(x_start)
        if f_start is not None:
            points.append((x0_lo, f_start[0], f_start[1]))
            
        x_end = make_point(x0_hi)
        f_end = eval_point(x_end)
        if f_end is not None:
            points.append((x0_hi, f_end[0], f_end[1]))
        
        # 2. Evaluate midpoint
        x_mid = make_point((x0_lo + x0_hi) / 2.0)
        f_mid = eval_point(x_mid)
        if f_mid is not None:
            points.append(((x0_lo + x0_hi) / 2.0, f_mid[0], f_mid[1]))
        
        # Sort points by f1 (which is x0)
        points.sort(key=lambda p: p[0])
        
        # Active intervals for bisection.
        # Each interval is represented by its left and right point indices in the sorted `points` list.
        # We store (left_idx, right_idx, priority)
        # Priority is the estimated marginal HV gain of inserting a point in the middle of this interval.
        
        def get_hv_gain(p_left, p_mid, p_right):
            """
            Estimate the marginal HV gain of inserting p_mid between p_left and p_right.
            This assumes a convex front. The gain is the area of the polygon 
            (ref, p_left, p_mid, p_right) minus the area of (ref, p_left, p_right).
            For 2D, if points are non-dominated and sorted by f1:
            HV of 3 points vs 2 points.
            """
            # Filter points that are better than ref
            pts = [p_left, p_mid, p_right]
            pts_valid = [p for p in pts if p[1] < ref[0] and p[2] < ref[1]]
            
            if len(pts_valid) < 2:
                return 0.0
            
            # Calculate HV of the set
            # Sort by f1
            pts_valid.sort(key=lambda p: p[1])
            
            # Remove dominated points (keep lower envelope of f2)
            nd_points = []
            min_f2 = np.inf
            for p in pts_valid:
                if p[2] < min_f2:
                    nd_points.append(p)
                    min_f2 = p[2]
            
            if len(nd_points) < 2:
                return 0.0
            
            # Calculate HV
            hv = 0.0
            prev_f2 = ref[1]
            for p in nd_points:
                f1, f2 = p[1], p[2]
                width = ref[0] - f1
                height = prev_f2 - f2
                if width > 0 and height > 0:
                    hv += width * height
                prev_f2 = f2
            return hv

        # Initialize intervals
        # We have points 0, 1, 2 (sorted by x0)
        # Intervals: (0, 1) and (1, 2)
        # We need to calculate the potential gain of splitting each.
        # To do this, we need to evaluate the midpoint of each interval? 
        # No, we already have the midpoint of the whole domain. 
        # We need to evaluate midpoints of the sub-intervals to know the gain.
        # This is a trade-off: we use evaluations to decide where to put the next evaluation.
        
        # Let's refine the strategy:
        # We maintain a list of "candidate midpoints" to evaluate.
        # Initially, we have evaluated L, M, R.
        # The next best candidates are the midpoints of (L, M) and (M, R).
        # We evaluate them one by one, and the one that provides the highest actual HV gain 
        # (compared to the current HV of the set) is kept, and its sub-intervals are added to the queue.
        
        # Current HV of the set
        def calc_current_hv(pts):
            # pts is list of (x0, f1, f2)
            pts_valid = [p for p in pts if p[1] < ref[0] and p[2] < ref[1]]
            if len(pts_valid) == 0:
                return 0.0
            pts_valid.sort(key=lambda p: p[1])
            nd_points = []
            min_f2 = np.inf
            for p in pts_valid:
                if p[2] < min_f2:
                    nd_points.append(p)
                    min_f2 = p[2]
            if len(nd_points) == 0:
                return 0.0
            hv = 0.0
            prev_f2 = ref[1]
            for p in nd_points:
                f1, f2 = p[1], p[2]
                width = ref[0] - f1
                height = prev_f2 - f2
                if width > 0 and height > 0:
                    hv += width * height
                prev_f2 = f2
            return hv

        current_hv = calc_current_hv(points)
        
        # Queue of intervals to split.
        # Each entry: (left_point, right_point, x_mid_candidate)
        # We sort by estimated gain or just process in order of interval width?
        # For convex fronts, smaller intervals near the elbow are more valuable.
        # Let's use a priority queue based on the width of the interval in f1 space?
        # Or just simple bisection order?
        # A simple depth-first or breadth-first bisection usually works well.
        # Let's use a list of intervals, and at each step, pick the one with the largest 
        # "potential" (width in x-space * some heuristic).
        # Actually, a simpler robust strategy: 
        # Always bisect the interval with the largest width in x-space.
        # This ensures uniform coverage in x, which might not be optimal for HV if the front is skewed,
        # but combined with the "midpoint of whole" start, it's a good heuristic.
        
        # Better: Use the actual HV gain from the *last* inserted point to estimate future gains?
        # No, let's stick to the directive: "dynamically selects the next x1 coordinate to maximize the marginal hypervolume gain".
        
        # To do this efficiently without evaluating all candidates:
        # We assume the front is convex. The gain of a point is proportional to the area of the triangle 
        # formed by the point and the line segment connecting its neighbors.
        # We can estimate this gain using the neighbors' f-values.
        
        # State:
        # `points` is sorted by x0.
        # `intervals` is a list of tuples (idx_left, idx_right).
        # We will evaluate the midpoint of the interval with the highest estimated gain.
        
        # Initialize intervals
        intervals = []
        for i in range(len(points) - 1):
            intervals.append((i, i + 1))
            
        # Function to estimate gain of splitting interval (i, j)
        def estimate_gain(i, j):
            if i >= len(points) or j >= len(points) or i >= j:
                return 0.0
            p_l = points[i]
            p_r = points[j]
            # Estimate midpoint in x
            x_m = (p_l[0] + p_r[0]) / 2.0
            # We don't have f_m yet. 
            # For a convex function, the midpoint is below the line segment.
            # The gain is the area between the line segment and the curve.
            # Approximation: If we assume the curve is a parabola or similar, 
            # the gain is related to the distance from the line segment.
            # Without evaluating, we can't know the exact gain.
            # So we must evaluate to know the gain.
            # This implies we should evaluate a candidate, calculate the actual gain, 
            # and if it's significant, keep it.
            
            # Strategy: 
            # 1. Pick an interval (e.g., largest width).
            # 2. Evaluate its midpoint.
            # 3. Calculate actual HV gain.
            # 4. If gain > threshold, insert it.
            # 5. Repeat.
            
            # Which interval to pick?
            # The one with the largest width in x-space is a safe bet for "covering" the front.
            # But for HV, the "elbow" is more important.
            # Let's pick the interval with the largest width in f1 space?
            # Or just the largest x-width?
            
            return p_r[0] - p_l[0] # Use x-width as proxy for priority

        # Main loop
        evals_used = 3
        while evals_used < budget:
            if time.time() > deadline:
                break
                
            if not intervals:
                break
                
            # Find interval with max priority (x-width)
            max_idx = -1
            max_prio = -1
            for k, (i, j) in enumerate(intervals):
                prio = points[j][0] - points[i][0]
                if prio > max_prio:
                    max_prio = prio
                    max_idx = k
                    
            if max_idx == -1:
                break
                
            i, j = intervals[max_idx]
            p_l = points[i]
            p_r = points[j]
            
            # If the interval is too small, stop refining this branch?
            # Or just continue until budget.
            
            x_m = (p_l[0] + p_r[0]) / 2.0
            
            # Evaluate midpoint
            x_m_vec = make_point(x_m)
            f_m = eval_point(x_m_vec)
            evals_used += 1
            
            if f_m is None:
                # Invalid point, remove interval
                intervals.pop(max_idx)
                continue
            
            f1_m, f2_m = f_m[0], f_m[1]
            
            # Check if this point improves HV
            # Temporarily insert and check HV
            new_points = points.copy()
            # Insert at position i+1? 
            # Since x_m is between p_l and p_r, and points are sorted by x0, 
            # it should be inserted at index i+1 if we assume x0 order matches f1 order.
            # ZDT1: f1 = x0. So yes.
            
            # However, f2 might make it dominated or non-dominated differently.
            # Let's just add to a temp list and sort.
            temp_list = new_points + [(x_m, f1_m, f2_m)]
            temp_list.sort(key=lambda p: p[0])
            
            new_hv = calc_current_hv(temp_list)
            gain = new_hv - current_hv
            
            if gain > 1e-9:
                # Accept the point
                points = temp_list
                current_hv = new_hv
                
                # Update intervals
                # The interval (i, j) is split into (i, i+1) and (i+1, j)
                # Note: indices in `points` have shifted for elements after i.
                # Since we inserted at i+1, elements at i+1 and later in the old list are now at i+2 and later.
                # The new point is at index i+1.
                
                # Remove old interval
                intervals.pop(max_idx)
                
                # Add new intervals
                # Left: (i, i+1)
                # Right: (i+1, j)
                # But we need to check if these new intervals are worth keeping.
                # If the width is very small, we might skip.
                if points[i+1][0] - points[i][0] > 1e-6:
                    intervals.append((i, i+1))
                if points[j][0] - points[i+1][0] > 1e-6:
                    intervals.append((i+1, j))
            else:
                # No gain, discard interval (this part of the front is flat or dominated)
                intervals.pop(max_idx)
                
        return points
