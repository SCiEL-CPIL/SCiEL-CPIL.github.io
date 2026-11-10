# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    # Precompute for fast evaluation
    def check_inside(x, y, a_deg):
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        wx = X_flat - x
        wy = Y_flat - y
        
        u = wx * c - wy * s
        v = wx * s + wy * c
        
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        return in_r1 | in_r2
    
    def evaluate_path(path):
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        mask = np.ones(n_points, dtype=bool)
        
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            ts = np.linspace(0, 1, m+1)
            
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                
                inside = check_inside(x, y, a)
                mask &= inside
                
                if not mask.any():
                    return 0.0
        
        return np.sum(mask) * 0.0001
    
    # Vectorized evaluation: compute all placements at once
    def evaluate_path_vectorized(path):
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # Collect all (x, y, a) placements
        placements = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            ts = np.linspace(0, 1, m+1)
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append((x, y, a))
        
        if len(placements) == 0:
            return float(n_points) * 0.0001
        
        # Vectorize: for each placement, compute which points are inside
        # We'll process in batches to avoid memory issues
        mask = np.ones(n_points, dtype=bool)
        
        batch_size = 20
        for start in range(0, len(placements), batch_size):
            end = min(start + batch_size, len(placements))
            batch = placements[start:end]
            
            for x, y, a in batch:
                theta = -np.deg2rad(a)
                c = np.cos(theta)
                s = np.sin(theta)
                
                wx = X_flat - x
                wy = Y_flat - y
                
                u = wx * c - wy * s
                v = wx * s + wy * c
                
                in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
                in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
                
                inside = in_r1 | in_r2
                mask &= inside
                
                if not mask.any():
                    return 0.0
        
        return np.sum(mask) * 0.0001
    
    # Smooth surrogate using soft-min
    def soft_min_surrogate(path, beta=20.0):
        """
        Smooth approximation of the min over all placements of the indicator.
        Uses log-sum-exp trick for numerical stability.
        """
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # Collect all placements
        placements = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            ts = np.linspace(0, 1, m+1)
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append((x, y, a))
        
        if len(placements) == 0:
            return float(n_points)
        
        # For each point, compute the minimum "distance" to being outside
        # across all placements. Use a smooth approximation.
        
        # For each placement, compute a "score" for each point:
        # 1 if inside, 0 if outside. We want the min over placements.
        # Soft min: -log(sum(exp(-beta * score_i))) / beta
        # But score is binary, so let's use a different approach.
        
        # Alternative: for each point, compute the minimum "margin" of being inside.
        # For a point to be inside placement (x,y,a), it must satisfy:
        # (0<=u<=1, -3<=v<=1) or (-3<=u<=1, 0<=v<=1)
        # The margin is the minimum distance to the boundary of the L-shape.
        
        # This is complex. Let's use a simpler approach:
        # Compute the product (or sum of logs) of indicators, which approximates AND.
        
        # For numerical stability, we'll compute the sum of "violations"
        # and use a soft penalty.
        
        total_violation = np.zeros(n_points)
        
        batch_size = 20
        for start in range(0, len(placements), batch_size):
            end = min(start + batch_size, len(placements))
            batch = placements[start:end]
            
            for x, y, a in batch:
                theta = -np.deg2rad(a)
                c = np.cos(theta)
                s = np.sin(theta)
                
                wx = X_flat - x
                wy = Y_flat - y
                
                u = wx * c - wy * s
                v = wx * s + wy * c
                
                # Compute violation: how far outside the L-shape
                # R1: 0<=u<=1, -3<=v<=1
                # R2: -3<=u<=1, 0<=v<=1
                # A point is inside if it's in R1 or R2.
                # Violation for R1: max(0, -u, u-1, v+3, v-1)
                # Violation for R2: max(0, u+3, u-1, -v, v-1)
                # Total violation: min(violation_R1, violation_R2)
                
                viol_r1 = np.maximum.reduce([
                    np.zeros_like(u),
                    -u,
                    u - 1.0,
                    v + 3.0,
                    v - 1.0
                ])
                
                viol_r2 = np.maximum.reduce([
                    np.zeros_like(u),
                    u + 3.0,
                    u - 1.0,
                    -v,
                    v - 1.0
                ])
                
                viol = np.minimum(viol_r1, viol_r2)
                
                # Soft penalty: exp(-beta * viol) approximates indicator
                # Sum of violations (soft)
                total_violation += viol
        
        # The sofa points are those with total_violation close to 0
        # Score approximation: count points with small violation
        # Use a smooth count: sum(exp(-beta * total_violation))
        beta_local = 50.0
        scores = np.exp(-beta_local * total_violation)
        
        return np.sum(scores)
    
    # Build a path analytically based on the sofa geometry
    # The sofa is 4.5 long and 3 wide
    # We need to move it from corridor 1 (horizontal, extending right from origin)
    # to corridor 2 (vertical, extending up from (-2,0))
    
    # Key insight: The sofa needs to pivot around the inner corner
    # At angle 0: sofa is in horizontal corridor, inner corner at (x,y)
    # At angle 90: sofa is in vertical corridor, inner corner at (-2,0)
    
    # The critical constraint is that the sofa must fit in the L-shaped corridor
    # at all times. The corridor has width 1.
    
    # Let's build a smooth path with key waypoints
    # The sofa's long axis rotates from 0 to 90 degrees
    
    # Analytical path: the inner corner traces a curve that keeps the sofa
    # tangent to the outer walls of the L-corridor
    
    # For a sofa of length L=4.5 and width W=3, moving through an L-corridor
    # of width 1, the inner corner path is complex but we can approximate
    
    # Key positions:
    # a=0: sofa in horizontal arm, inner corner at (0,0) [fixed by evaluator]
    # a=90: sofa in vertical arm, inner corner at (-2,0) [fixed by evaluator]
    
    # The sofa has to "turn the corner" - this is the classic sofa problem
    
    # Let's use a path where the sofa slides along the outer wall then pivots
    # The critical region is when the sofa is at ~45 degrees
    
    # Build path with multiple segments
    # Segment 1: slide right while starting to rotate (a: 0 -> 30)
    # Segment 2: pivot through the corner (a: 30 -> 60)  
    # Segment 3: slide down while finishing rotation (a: 60 -> 90)
    
    path = []
    
    # a=0: start (0,0,0) - handled by evaluator
    # First intermediate: slide out a bit and start rotating
    # At a=15, the sofa should be mostly in the horizontal corridor
    # Inner corner position: need to keep sofa in corridor
    # The sofa extends 4.5 in one direction and 3 in perpendicular
    # When rotated by a, the projection changes
    
    # For small angles, the sofa fits if we move the corner appropriately
    # The sofa's far end is at distance 4.5 along its length from corner
    # At angle a, the far end is at (x + 4.5*cos(a), y + 4.5*sin(a)) relative to corner
    # Wait, the base corridor has the sofa with inner corner at origin
    # The sofa extends from u=0 to u=1 (width 1 in the "width" direction)
    # and u=-3 to u=1 (length 4 in the "length" direction)
    # Actually the base is:
    # R1: 0 <= u <= 1, -3 <= v <= 1  (the "width" part, 1 wide, 4 long)
    # R2: -3 <= u <= 1, 0 <= v <= 1  (the "length" part, 4 long, 1 wide)
    # So the sofa is an L-shape? No wait...
    # R1 is a rectangle from (0,-3) to (1,1) - 1 unit wide, 4 units tall
    # R2 is a rectangle from (-3,0) to (1,1) - 4 units wide, 1 unit tall
    # Their union is an L-shape with the corner at (0,0) to (1,1)
    
    # So the "sofa" is actually the L-shaped corridor itself!
    # The sofa IS the corridor shape. We're moving the corridor through itself.
    
    # This means we're rotating an L-shape and checking if it still fits
    # in the original L-shaped region.
    
    # The L-shape has arms of length 4 and width 1, meeting at a 1x1 square
    # We rotate it about its inner corner (0,0 in local coords)
    
    # For the L to fit in itself when rotated by angle a about (x,y):
    # The rotated L must be a subset of the original L region
    
    # This is a self-rotation problem. The maximum rotation before it doesn't
    # fit is the key constraint.
    
    # For a 90 degree rotation, the L maps to itself (the horizontal arm becomes
    # vertical and vice versa), but the position shifts.
    
    # At a=0: L at (0,0) fits in L at (0,0) - trivially
    # At a=90: L rotated 90 deg, corner at (-2,0)
    # The rotated L: arm 1 was vertical (u:0-1, v:-3-1), now horizontal
    # arm 2 was horizontal (u:-3-1, v:0-1), now vertical
    # After 90 deg rotation: arm 1 becomes u:-3-1, v:0-1 -> wait
    # R(-90)(u,v) = (v, -u)
    # R1: u in [0,1], v in [-3,1] -> new coords: v in [-3,1] becomes x, -u in [-1,0] becomes y
    # So R1 maps to x in [-3,1], y in [-1,0]
    # R2: u in [-3,1], v in [0,1] -> x in [0,1], y in [-1,3]
    # Union: x in [-3,1], y in [-1,0] union x in [0,1], y in [-1,3]
    # This is an L-shape with corner at (0,0) to (1,0), arms extending left and up
    
    # Placed at (-2,0): the corner is at (-2,0)
    # R1 maps to: x in [-5,-1], y in [-1,0] relative to (-2,0) -> absolute x in [-5,-1], y in [-1,0]
    # Wait, that's outside the region.
    
    # Let me reconsider. The placement (x,y,a) means: point w is inside if R(-a)(w-(x,y)) is in base.
    # So the sofa shape is the base rotated by a and translated to (x,y).
    # The sofa must be inside the "corridor" which is... the fixed region?
    
    # Re-reading: "Row (x, y, a) places the width-1 L-shaped corridor with its inner corner at (x, y), rotated by a degrees."
    # "The sofa is the set of grid points ... that are inside every placement"
    
    # So the sofa points must be inside EVERY placement along the path.
    # The "corridor" is the L-shaped region, and we're placing it at different positions/orientations.
    # The sofa is the intersection of all these placements.
    
    # So we need to find a path of (x,y,a) such that the intersection of all the
    # L-shaped regions (rotated and translated) is as large as possible.
    
    # The L-shaped region at (x,y,a) is: {w : R(-a)(w-(x,y)) in base_L}
    
    # For a=0: base_L at (0,0) is the L from (-3,-1) to (1,1) in the corner
    # For a=90 at (-2,0): the L rotated 90 deg, placed at (-2,0)
    
    # The sofa is the points that are in ALL these L-regions.
    
    # To maximize the sofa, we want the L-regions to overlap as much as possible.
    # This means we want the path to keep the L-region roughly in the same place.
    
    # The constraint is that the path starts at (0,0,0) and ends at (-2,0,90).
    # We need to find intermediate positions where the L-region still overlaps
    # with the start and end positions as much as possible.
    
    # Key insight: The L-region has a specific shape. When we rotate it 90 degrees,
    # it maps the horizontal arm to vertical and vice versa. The overlap between
    # the a=0 and a=90 positions is limited.
    
    # Let's compute the overlap between:
    # Start: L at (0,0,0)
    # End: L at (-2,0,90)
    
    # Start L: {w: w-(0,0) in base} = base = {(u,v): (0<=u<=1, -3<=v<=1) or (-3<=u<=1, 0<=v<=1)}
    # In world coords (since a=0, x=0, y=0): same as base
    # So start region: x in [0,1], y in [-3,1] union x in [-3,1], y in [0,1]
    
    # End: L at (-2,0,90). w is inside if R(-90)(w-(-2,0)) in base.
    # R(-90)(p) = (p_y, -p_x) for point p
    # w-(-2,0) = (w_x+2, w_y)
    # R(-90)(w_x+2, w_y) = (w_y, -(w_x+2))
    # This must be in base: (0<=u<=1, -3<=v<=1) or (-3<=u<=1, 0<=v<=1)
    # So: (0<=w_y<=1, -3<=-(w_x+2)<=1) or (-3<=w_y<=1, 0<=-(w_x+2)<=1)
    # First: 0<=w_y<=1 and -1<=w_x+2<=3 -> 0<=w_y<=1 and -3<=w_x<=1
    # Second: -3<=w_y<=1 and -1<=w_x+2<=0 -> -3<=w_y<=1 and -3<=w_x<=-2
    # Union: (0<=w_y<=1, -3<=w_x<=1) union (-3<=w_y<=1, -3<=w_x<=-2)
    
    # Overlap of start and end:
    # Start: (0<=x<=1, -3<=y<=1) union (-3<=x<=1, 0<=y<=1)
    # End: (0<=y<=1, -3<=x<=1) union (-3<=y<=1, -3<=x<=-2)
    
    # The overlap is the intersection. Let's find it:
    # Start region 1: x in [0,1], y in [-3,1]
    # Start region 2: x in [-3,1], y in [0,1]
    # End region 1: x in [-3,1], y in [0,1]  (since 0<=y<=1, -3<=x<=1)
    # End region 2: x in [-3,-2], y in [-3,1]
    
    # Start1 ∩ End1: x in [0,1], y in [0,1]
    # Start1 ∩ End2: x in [0,1] ∩ [-3,-2] = empty
    # Start2 ∩ End1: x in [-3,1] ∩ [-3,1] = [-3,1], y in [0,1] ∩ [0,1] = [0,1] -> x in [-3,1], y in [0,1]
    # Wait, Start2 is x in [-3,1], y in [0,1], End1 is x in [-3,1], y in [0,1]
    # So Start2 ∩ End1 = x in [-3,1], y in [0,1]
    
    # Start2 ∩ End2: x in [-3,1] ∩ [-3,-2] = [-3,-2], y in [0,1] ∩ [-3,1] = [0,1]
    
    # Total overlap: union of (x in [0,1], y in [0,1]) and (x in [-3,1], y in [0,1]) and (x in [-3,-2], y in [0,1])
    # = x in [-3,1], y in [0,1]
    
    # So the maximum possible sofa (if we only had start and end) is x in [-3,1], y in [0,1]
    # Area = 4 * 1 = 4 square units
    # In grid points: 400 * 100 = 40,000 points
    # Score = 40000 * 0.0001 = 4.0
    
    # But we also have intermediate positions, which further constrain the sofa.
    # The sofa must be inside ALL placements, so each intermediate placement
    # can only reduce the sofa further.
    
    # To maximize the sofa, we want intermediate placements that are as "close"
    # to the start/end as possible, so they don't cut away much.
    
    # The key question: can we go from (0,0,0) to (-2,0,90) with intermediate
    # placements that all contain the region x in [-3,1], y in [0,1]?
    
    # If yes, the sofa is at least that region, and the score is at least 4.0.
    
    # Let's check: for a point (x,y) in [-3,1] x [0,1], is it inside the L at
    # position (px, py, pa) for some path from (0,0,0) to (-2,0,90)?
    
    # We need to find a path such that for all (px,py,pa) on the path,
    # the region x in [-3,1], y in [0,1] is inside the L at (px,py,pa).
    
    # This is a strong constraint. Let's think about what L-regions contain
    # the square [-3,1] x [0,1].
    
    # The L at (px,py,pa) contains (px,py) region rotated. For the square to be
    # inside, the L must cover it.
    
    # This is complex. Let's just use a numerical approach with a well-designed
    # initial path and then optimize.
    
    # Based on the analysis, the sofa is likely to be concentrated in the region
    # x in [-3,1], y in [0,1] or a subset thereof.
    
    # Let's build a path that keeps the L-region covering this region as much as possible.
    
    # Path strategy: 
    # The L-region at angle a has its "corner" at (x,y). The arms extend in directions
    # rotated by a from the base directions.
    
    # For the square [-3,1]x[0,1] to be covered, the L needs to span this region.
    
    # Let's just use a smooth path and optimize numerically.
    
    # Initial path: interpolate from (0,0,0) to (-2,0,90) with some control points
    # The angle goes from 0 to 90, so we need to rotate.
    
    # A simple path: linear interpolation of (x,y,a)
    # x(t) = 0 + t*(-2) = -2t
    # y(t) = 0
    # a(t) = 90t
    
    # This gives a path where the corner moves left and the angle increases.
    
    # Let's try this as a starting point and then optimize.
    
    def make_simple_path(n_points=10):
        t = np.linspace(0.1, 0.9, n_points)  # avoid endpoints (handled by evaluator)
        path = []
        for ti in t:
            x = -2.0 * ti
            y = 0.0
            a = 90.0 * ti
            path.append([x, y, a])
        return np.array(path, dtype=np.float64)
    
    # Try the simple linear path first
    simple_path = make_simple_path(8)
    simple_score = evaluate_path(simple_path)
    
    # Now try to optimize with more control points
    # Use 6 control points at specific angles
    angles = [15.0, 30.0, 45.0, 60.0, 75.0, 85.0]
    
    def make_path_from_params(params):
        path = []
        for i, a in enumerate(angles):
            path.append([params[2*i], params[2*i+1], a])
        return np.array(path, dtype=np.float64)
    
    # Initial parameters: linear interpolation
    current_params = np.array([
        -2.0*15.0/90.0, 0.0,   # a=15
        -2.0*30.0/90.0, 0.0,   # a=30
        -2.0*45.0/90.0, 0.0,   # a=45
        -2.0*60.0/90.0, 0.0,   # a=60
        -2.0*75.0/90.0, 0.0,   # a=75
        -2.0*85.0/90.0, 0.0,   # a=85
    ])
    
    best_path = simple_path
    best_score = simple_score
    
    # Also try the seed path from the original
    seed_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    seed_score = evaluate_path(seed_path)
    if seed_score > best_score:
        best_score = seed_score
        best_path = seed_path
    
    start_time = time.time()
    time_limit = 580.0
    
    # Nelder-Mead optimization
    def nelder_mead(f, x0, max_iter=200, tol=1e-6, time_limit_check=None):
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        simplex = [x0.copy()]
        fvals = [f(x0)]
        
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.1 if x0[i] != 0 else 0.1
            simplex.append(xi)
            fvals.append(f(xi))
        
        best_params = x0.copy()
        best_val = min(fvals)
        
        for iteration in range(max_iter):
            if time_limit_check is not None and time.time() > time_limit_check:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            
            if fvals[0] < best_val:
                best_val = fvals[0]
                best_params = simplex[0].copy()
        
        return best_params, best_val
    
    # Objective using exact evaluation
    def objective_exact(params):
        path = make_path_from_params(params)
        return -evaluate_path(path)
    
    # Objective using soft surrogate (faster, smoother)
    def objective_soft(params, beta=20.0):
        path = make_path_from_params(params)
        return -soft_min_surrogate(path, beta)
    
    # Run Nelder-Mead with exact objective first
    deadline = start_time + time_limit
    np_params, np_val = nelder_mead(
        objective_exact, current_params,
        max_iter=150, tol=1e-7,
        time_limit_check=deadline
    )
    
    if -np_val > best_score:
        best_score = -np_val
        best_path = make_path_from_params(np_params)
    
    # Try random restarts
    np.random.seed(42)
    for restart in range(8):
        if time.time() > deadline - 30:
            break
        
        perturbed = current_params + np.random.normal(0, 0.15, size=len(current_params))
        bp, bv = nelder_mead(
            objective_exact, perturbed,
            max_iter=60, tol=1e-6,
            time_limit_check=deadline
        )
        
        if -bv > best_score:
            best_score = -bv
            best_path = make_path_from_params(bp)
    
    # Final polish: use the soft surrogate to guide a local search, then verify with exact
    if time.time() < deadline - 15:
        # Use best path as starting point
        polish_params = np.array([
            best_path[0, 0], best_path[0, 1],
            best_path[1, 0], best_path[1, 1],
            best_path[2, 0], best_path[2, 1],
            best_path[3, 0], best_path[3, 1],
            best_path[4, 0], best_path[4, 1],
            best_path[5, 0], best_path[5, 1],
        ]) if len(best_path) >= 6 else current_params
        
        # Polish with tighter tolerance
        pp, pv = nelder_mead(
            objective_exact, polish_params,
            max_iter=100, tol=1e-8,
            time_limit_check=deadline
        )
        
        if -pv > best_score:
            best_score = -pv
            best_path = make_path_from_params(pp)
    
    # Safety: ensure we return a valid path
    if len(best_path) < 1:
        best_path = make_simple_path(5)
    
    # Check all values are finite
    if not np.all(np.isfinite(best_path)):
        best_path = make_simple_path(5)
    
    return best_path

# EVOLVE-BLOCK-END
