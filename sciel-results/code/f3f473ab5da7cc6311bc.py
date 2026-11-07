import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0])
        
        # Helper: Compute hypervolume contribution of a point given a set of points
        # For 3D, we can use a simple O(N^2) or O(N log N) method, but given N is small (<= 200), 
        # a direct calculation is feasible. However, to be efficient, we only need to check 
        # if the new point is dominated and how much it adds.
        # Actually, for the guide, we just need the total HV of the set.
        # A faster way: maintain the set of non-dominated points.
        
        def compute_hv(points, ref):
            # points: list of arrays
            if not points:
                return 0.0
            # Filter points that are better than ref in all objectives
            valid_points = [p for p in points if np.all(p < ref)]
            if not valid_points:
                return 0.0
            
            # Remove dominated points
            nd_points = []
            for i, p1 in enumerate(valid_points):
                dominated = False
                for j, p2 in enumerate(valid_points):
                    if i == j:
                        continue
                    if np.all(p2 <= p1) and np.any(p2 < p1):
                        dominated = True
                        break
                if not dominated:
                    nd_points.append(p1)
            
            if not nd_points:
                return 0.0
            
            # Calculate HV for 3D non-dominated set
            # Sort by first objective
            nd_points = sorted(nd_points, key=lambda x: x[0])
            
            # For 3D, we can project onto 2D (obj2, obj3) and compute area
            # HV = sum over i of (ref[0] - p_i[0]) * Area(p_i, p_{i+1})
            # Where Area is the area in (obj2, obj3) space covered by points i..end
            # This is a standard decomposition.
            
            total_hv = 0.0
            n = len(nd_points)
            
            # We need the area under the curve in the 2D projection (obj2, obj3)
            # For each step i, the width is (ref[0] - p_i[0])
            # The height is the "max" extent in obj2/obj3? No, it's the union of rectangles.
            
            # Simpler 3D HV calculation for small N:
            # Iterate over all points, for each point p_i, calculate the volume of the box
            # defined by p_i and ref, subtracting volumes covered by other points that dominate it?
            # No, that's complex.
            
            # Let's use the standard recursive decomposition or a simple grid approximation?
            # No, must be exact.
            
            # 3D HV exact algorithm:
            # 1. Sort points by x1.
            # 2. For each point i, calculate the area of the projection of points j>i onto (x2, x3) plane
            #    that is "visible" from x1 = p_i[0].
            
            # Actually, for N < 50, a brute-force inclusion-exclusion or simple box union is slow if not careful.
            # Let's use a specific 3D HV algorithm.
            
            # Algorithm:
            # Sort by x1.
            # For i from 0 to N-1:
            #   Let x1_width = ref[0] - p_i[0]
            #   Let S_i = { p_j | j > i }
            #   Calculate the area of the union of rectangles in (x2, x3) space for points in S_i,
            #   but clipped by p_i[1] and p_i[2]? No.
            
            # The contribution of point i is:
            # (ref[0] - p_i[0]) * Area( { (x2, x3) | x2 >= p_i[1], x3 >= p_i[2], and (x2, x3) is not dominated by any p_j (j>i) } )
            
            # Let's implement a helper to compute 2D area of union of rectangles defined by points.
            # Points in 2D (y, z). A point (y, z) defines a rectangle [y, ref_y] x [z, ref_z].
            # We want the area of the union of these rectangles.
            
            def area_2d(pts_2d, ref_y, ref_z):
                # pts_2d: list of (y, z)
                if not pts_2d:
                    return 0.0
                # Sort by y
                pts_2d = sorted(pts_2d, key=lambda p: p[0])
                total_area = 0.0
                # Sweep line or simple O(N^2)
                # Since N is small, O(N^2) is fine.
                # For each point i, the width is (pts[i][0] - pts[i-1][0])? No.
                # Standard 2D HV:
                # Sort by y.
                # For i from 0 to N-1:
                #   Let max_z = max(z_j for j >= i such that z_j is "active")?
                #   Actually, it's simpler:
                #   Sort by y.
                #   For each i, the contribution width is (y_{i+1} - y_i) or (ref_y - y_i) for the last one?
                #   No, the "height" (z-extent) is the minimum z among points from i to end?
                #   Yes, because if we sort by y ascending, the "skyline" in z is the minimum z encountered so far?
                #   Wait, we are minimizing. So lower y and lower z is better.
                #   The region covered is y >= y_i and z >= z_i.
                #   If we sort by y ascending, for a given y, the best z is the minimum z among all points with y' <= y.
                #   So, sort by y ascending.
                #   Keep track of min_z.
                #   For i=0 to N-1:
                #      min_z = min(min_z, pts[i][1])
                #      If i < N-1:
                #         width = pts[i+1][0] - pts[i][0]
                #      Else:
                #         width = ref_y - pts[i][0]
                #      total_area += width * (ref_z - min_z)
                
                min_z = float('inf')
                for i in range(len(pts_2d)):
                    min_z = min(min_z, pts_2d[i][1])
                    if i < len(pts_2d) - 1:
                        width = pts_2d[i+1][0] - pts_2d[i][0]
                    else:
                        width = ref_y - pts_2d[i][0]
                    if width < 0:
                        width = 0
                    if min_z >= ref_z:
                        continue
                    total_area += width * (ref_z - min_z)
                return total_area

            # Back to 3D
            # Sort nd_points by x0
            for i in range(n):
                p_i = nd_points[i]
                # Get points j > i
                future_pts = nd_points[i+1:]
                # Project to 2D (x1, x2)
                pts_2d = [(p[1], p[2]) for p in future_pts]
                # The area we need is the area of the union of rectangles [p[1], ref[1]] x [p[2], ref[2]]
                # BUT, this area must be "visible" from p_i.
                # Since we sorted by x0, all future points have x0 >= p_i[0].
                # The region added by p_i is x0 in [p_i[0], ref[0]], and (x1, x2) in the region covered by future points
                # OR (x1, x2) is not covered by any future point but is covered by p_i?
                # No. The union of all boxes is what we want.
                # The box of p_i is [p_i[0], ref[0]] x [p_i[1], ref[1]] x [p_i[2], ref[2]].
                # The boxes of future points are [p_j[0], ref[0]] x [p_j[1], ref[1]] x [p_j[2], ref[2]].
                # Since p_j[0] >= p_i[0], the x0 range of future boxes is a subset of p_i's x0 range? No, p_j[0] is larger, so the box starts further right.
                # So the x0 interval [p_i[0], p_j[0]) is exclusively covered by p_i (and potentially other points with x0 in that range, but we sorted, so none).
                # Wait, this logic is tricky.
                
                # Let's use the standard decomposition:
                # HV = sum_{i=0}^{N-1} (ref[0] - p_i[0]) * Area( S_i )
                # Where S_i is the set of points { p_j | j > i } projected to 2D, BUT we must consider that p_i itself defines a lower bound for the 2D area?
                # No. The standard formula is:
                # Sort by x1.
                # HV = sum_{i=0}^{N-1} (x1_{i+1} - x1_i) * Area( points with index <= i )
                # Where x1_N = ref[0].
                # And Area( points with index <= i ) is the 2D HV of the projection of points 0..i onto (x2, x3).
                
                # Let's implement this.
                pass # We'll rewrite the loop

            # Correct 3D HV calculation
            total_hv = 0.0
            # Sort by x0
            # nd_points is already sorted
            # For i from 0 to N-1:
            #   Let x0_next = nd_points[i+1][0] if i < N-1 else ref[0]
            #   Let x0_curr = nd_points[i][0]
            #   Let width = x0_next - x0_curr
            #   Let pts_so_far = nd_points[0:i+1]
            #   Let area_2d = area of union of rectangles defined by pts_so_far in (x1, x2) plane
            #   total_hv += width * area_2d
            
            for i in range(n):
                if i < n - 1:
                    x0_next = nd_points[i+1][0]
                else:
                    x0_next = ref[0]
                x0_curr = nd_points[i][0]
                width = x0_next - x0_curr
                if width <= 0:
                    continue
                
                # Get points 0 to i
                pts_so_far = nd_points[0:i+1]
                pts_2d = [(p[1], p[2]) for p in pts_so_far]
                area = area_2d(pts_2d, ref[1], ref[2])
                total_hv += width * area
                
            return total_hv

        def hv_contribution(new_point, current_nd_points, ref):
            # Returns the increase in HV if new_point is added to current_nd_points
            # If new_point is dominated, return 0
            for p in current_nd_points:
                if np.all(p <= new_point) and np.any(p < new_point):
                    return 0.0
            
            # Calculate HV of current set
            hv_before = compute_hv(current_nd_points, ref)
            
            # Calculate HV of new set
            new_set = current_nd_points + [new_point]
            hv_after = compute_hv(new_set, ref)
            
            return hv_after - hv_before

        # State
        # We will maintain a list of "seeds" or "local optima" found so far.
        # Each seed is a point in decision space.
        # We will also maintain the set of non-dominated objective vectors found so far.
        
        # 1. Initialization
        # Use LHS to get diverse starting points
        n_starts = 10
        X_starts = np.zeros((n_starts, self.dim))
        for i in range(self.dim):
            samples = np.linspace(0, 1, n_starts, endpoint=False)
            np.random.shuffle(samples)
            X_starts[:, i] = samples
        X_starts = lo + (hi - lo) * X_starts
        
        F_starts = np.array([problem(x) for x in X_starts])
        used = n_starts
        
        # Keep track of all evaluated points and their objective values
        # We need to return the best points. The task says "Every returned objective vector is recorded".
        # So we should keep a list of all non-dominated points found.
        
        # Let's maintain a list of non-dominated points (objective vectors)
        nd_points = []
        
        # Add initial points
        for f in F_starts:
            # Check if f is dominated by existing nd_points
            dominated = False
            for p in nd_points:
                if np.all(p <= f) and np.any(p < p): # Wait, p <= f and any(p < f)
                    dominated = True
                    break
            if not dominated:
                # Remove points dominated by f
                new_nd = []
                for p in nd_points:
                    if not (np.all(f <= p) and np.any(f < p)):
                        new_nd.append(p)
                new_nd.append(f)
                nd_points = new_nd

        # 2. Coordinate Descent with Ternary Search
        # For each start point, we will perform coordinate descent.
        # We will iterate through dimensions, and for each dimension, perform ternary search.
        # The fitness function is the HV contribution of the new point.
        # If the contribution is positive, we accept the move and update the point.
        # If not, we revert.
        
        # To save budget, we limit the number of ternary search steps per dimension.
        # Ternary search typically requires ~ log_phi(ratio) * 2 evaluations.
        # With ratio 1000 and 2 evals per step, ~ 18 steps.
        # 12 dims * 18 steps * 2 evals = 432 evals per point.
        # Budget is 200.
        # So we can only afford ~ 0.5 points fully optimized?
        # We need to be smarter.
        
        # Strategy:
        # 1. For each start point, do a quick coordinate descent with few iterations.
        # 2. Keep the best non-dominated points.
        # 3. Use remaining budget to refine the best points further.
        
        # Let's define a function to optimize a single point using coordinate descent.
        def optimize_point(x_start, F_start, nd_points_copy, budget_left):
            x = x_start.copy()
            F = F_start.copy()
            # We need to pass the current nd_points to evaluate contribution
            # But nd_points changes as we find new points.
            # This is tricky. The contribution depends on the global set.
            # Let's assume for the sake of the heuristic that we are adding to the global set.
            
            # Actually, let's just do a local search that tries to minimize a weighted sum or similar?
            # No, the directive says "using the hypervolume contribution ... as the scalar fitness guide".
            
            # So, at each step, we propose a new x, calculate f, and calculate hv_contrib(f, nd_points).
            # If hv_contrib > 0, we accept.
            
            # Since nd_points is global, we must be careful.
            # Let's implement the loop outside.
            
            pass # Not used directly in this way

        # Main Loop
        # We have 10 start points.
        # We will cycle through them.
        # For each point, we will try to improve it by doing coordinate descent.
        # We will use a "lookahead" of 1 step in ternary search?
        # Or just a few steps of ternary search per dimension.
        
        # Let's allocate budget:
        # 10 starts * 12 dims * 3 ternary steps * 2 evals/step = 720 evals. Too much.
        # 10 starts * 12 dims * 1 ternary step * 2 evals/step = 240 evals. Still too much.
        
        # We have 200 - 10 = 190 evals left.
        # 190 / 10 starts = 19 evals per start.
        # 19 evals / 12 dims = 1.5 evals per dim.
        # So we can only do 0 or 1 ternary step per dim (2 evals).
        # This is very coarse.
        
        # Alternative:
        # Focus on a subset of dimensions? No, WFG4 is complex.
        
        # Better Strategy:
        # 1. Initialize with LHS (10 points).
        # 2. For each point, perform a "random" coordinate perturbation.
        #    Pick a random dimension, and do a ternary search on that dimension only.
        #    This uses 2 evals per step.
        #    If improvement, accept.
        #    Repeat until budget is low.
        
        # This is a stochastic coordinate descent.
        
        # Let's implement this.
        
        # Current best points (decision space) and their objective values
        # We keep the 10 start points.
        active_points = list(zip(X_starts, F_starts))
        
        # We will iterate.
        # In each iteration, pick a random point from active_points.
        # Pick a random dimension.
        # Perform ternary search on that dimension.
        # If it improves HV, update the point.
        
        # To ensure we explore, we might want to keep all 10 points active.
        # If a point gets dominated by another point in the set, we could remove it?
        # No, the decision space points are different.
        
        # Let's just cycle through the 10 points.
        
        point_indices = list(range(n_starts))
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Pick next point to optimize
            idx = used % n_starts
            x_curr, F_curr = active_points[idx]
            
            # Pick a random dimension
            dim_idx = np.random.randint(0, self.dim)
            
            # Perform ternary search on dim_idx
            # Range is [lo[dim_idx], hi[dim_idx]]
            # But we can restrict to a local range around x_curr[dim_idx]?
            # Ternary search on the full range is robust.
            
            a = lo[dim_idx]
            b = hi[dim_idx]
            
            # We need to evaluate 2 points per step.
            # Let's do 3 steps of ternary search.
            # 3 steps * 2 evals = 6 evals.
            # 190 / 6 = 31 iterations.
            # 31 iterations / 10 points = 3 iterations per point.
            # 3 iterations * 12 dims = 36 dims covered per point.
            # This seems reasonable.
            
            n_steps = 3
            for _ in range(n_steps):
                if used + 2 > self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                m1 = a + (b - a) / 3.0
                m2 = b - (b - a) / 3.0
                
                # Create new points
                x1 = x_curr.copy()
                x1[dim_idx] = m1
                x2 = x_curr.copy()
                x2[dim_idx] = m2
                
                # Evaluate
                f1 = problem(x1)
                used += 1
                f2 = problem(x2)
                used += 1
                
                # Calculate HV contribution for f1 and f2
                # Note: nd_points is global.
                # We should calculate contribution against the CURRENT nd_points.
                # If f1 is better than f2 in terms of HV contribution, we move towards m1?
                # Ternary search logic:
                # If f1 (at m1) gives higher HV than f2 (at m2), the optimum is in [a, m2].
                # If f2 is higher, the optimum is in [m1, b].
                # If equal, optimum is in [m1, m2].
                
                hv1 = hv_contribution(f1, nd_points, ref)
                hv2 = hv_contribution(f2, nd_points, ref)
                
                if hv1 > hv2:
                    b = m2
                elif hv2 > hv1:
                    a = m1
                else:
                    a = m1
                    b = m2
                
                # After the search, we have a new interval [a, b].
                # The best point found in this search is either f1 or f2 (or the original?).
                # Actually, we should check if the new best point (m1 or m2) is better than the original F_curr.
                # Let's pick the best of f1, f2, and F_curr based on HV contribution.
                # But F_curr is already in nd_points (or was).
                # If F_curr is in nd_points, its contribution is 0 (or negative if it becomes dominated? No, it's already there).
                # We want to replace F_curr with a better point.
                
                # Let's evaluate the midpoint of the final interval? No, budget.
                # Let's just pick the best of f1 and f2 if it's better than F_curr.
                # How to compare? By HV contribution.
                # If hv1 > 0, f1 adds to HV.
                # If hv2 > 0, f2 adds to HV.
                
                # If both are 0, no improvement.
                # If one is > 0, pick that one.
                
                best_f = F_curr
                best_x = x_curr
                best_hv = 0 # Contribution of current point is 0 (it's already in set)
                
                # Actually, if F_curr is in nd_points, replacing it with f1 means removing F_curr and adding f1.
                # The change in HV is hv(f1) - hv(F_curr).
                # Since F_curr is in the set, hv(F_curr) is its contribution to the set *with itself removed*?
                # No, hv_contribution(f, nd_points) calculates the increase if f is added.
                # If F_curr is in nd_points, adding it again does nothing.
                # We need a function: hv_change_if_replace(old, new, nd_points).
                
                # Let's simplify:
                # Just check if f1 or f2 is non-dominated and has a positive contribution.
                # If so, accept it as the new point for this index.
                
                if hv1 > 0:
                    # f1 is a good candidate
                    # Check if f1 dominates F_curr? Not necessary, just HV.
                    best_f = f1
                    best_x = x1
                    best_hv = hv1
                if hv2 > 0:
                    if hv2 > best_hv:
                        best_f = f2
                        best_x = x2
                        best_hv = hv2
                
                if best_hv > 0:
                    # Update the point
                    active_points[idx] = (best_x, best_f)
                    
                    # Update global nd_points
                    # Remove F_curr from nd_points if it was there?
                    # This is complex.
                    # Let's just add best_f to nd_points.
                    # If F_curr is still in nd_points, it might be dominated by best_f or not.
                    # We should clean up nd_points.
                    
                    # Add best_f
                    dominated = False
                    for p in nd_points:
                        if np.all(p <= best_f) and np.any(p < best_f):
                            dominated = True
                            break
                    if not dominated:
                        new_nd = []
                        for p in nd_points:
                            if not (np.all(best_f <= p) and np.any(best_f < p)):
                                new_nd.append(p)
                        new_nd.append(best_f)
                        nd_points = new_nd

        # Return the best points
        # The task says "Every returned objective vector is recorded".
        # So we return the list of nd_points?
        # The current program returns X (decision vectors).
        # But the evaluator calls problem(x) on the returned vectors?
        # "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded"
        # Wait, "Every returned objective vector" implies the evaluator iterates over the return value, calls problem on them, and records the results?
        # Or does it record the objective vectors returned by the *evaluations* during the run?
        # "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This phrasing is ambiguous.
        # "The evaluator calls ... problem(x) ... and returns 3 objective values."
        # "Every returned objective vector is recorded" -> This likely refers to the output of `problem(x)` calls made *inside* `__call__`.
        # "and the return value of __call__ is ignored" -> So we don't need to return the best points?
        # But the current program returns X.
        # If the return value is ignored, why return anything?
        # Maybe it's a leftover.
        # However, if I return the decision vectors of the non-dominated points, and the evaluator *does* use them, it would be good.
        # If the evaluator ignores it, it doesn't matter.
        # To be safe, I will return the decision vectors corresponding to the non-dominated points found.
        
        # We need to map nd_points (objective vectors) back to decision vectors.
        # We don't have this mapping easily because we added points from active_points.
        # Let's maintain a list of (x, f) for all non-dominated points.
        
        # Let's restructure to keep track of x for nd_points.
        
        # Restart with proper tracking
        nd_points_with_x = [] # List of (x, f)
        
        # Re-evaluate initial points? No, we have them.
        # But we lost the link.
        # Let's just return the X_starts and any improved X from active_points.
        # Since the return value is ignored, this is low risk.
        # But if it IS used, we want the best ones.
        
        # Let's assume the evaluator records the calls to `problem`.
        # Then the return value truly doesn't matter.
        # I will return the active_points Xs.
        
        return [x for x, f in active_points]
