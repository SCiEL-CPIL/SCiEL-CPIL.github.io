import numpy as np
from scipy.optimize import differential_evolution

def moving_sofa_2d() -> np.ndarray:
    # Define the sofa points
    # i = 0..449, j = 0..299
    # x_sofa = -3 + 0.01*i
    # y_sofa = -1.5 + 0.01*j
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Pre-compute base corridor check function
    # Base: {0 <= u <= 1, -3 <= v <= 1} U {-3 <= u <= 1, 0 <= v <= 1}
    def inside_base(points):
        u = points[:, 0]
        v = points[:, 1]
        mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return mask1 | mask2

    def evaluate_path_params(params):
        # params: x1, y1, x2, y2, x3, y3, x4, y4
        # Keyframes at angles 0, 22.5, 45, 67.5, 90
        # Start (0,0,0), End (-2,0,90) are fixed by evaluator, but we need to include them in interpolation logic
        # Actually, the evaluator PREPENDS (0,0,0) and APPENDS (-2,0,90).
        # So our returned path is the intermediate points.
        # Let's define our keyframes to be at specific angles.
        # If we return a path, the evaluator adds (0,0,0) at start and (-2,0,90) at end.
        # To control the path well, we should probably specify points at regular angle intervals.
        # Let's assume we return points at angles 22.5, 45, 67.5.
        # Then the evaluator sees: (0,0,0), (x1,y1,22.5), (x2,y2,45), (x3,y3,67.5), (-2,0,90).
        
        x1, y1, x2, y2, x3, y3 = params
        
        keyframes = np.array([
            [0.0, 0.0, 0.0],
            [x1, y1, 22.5],
            [x2, y2, 45.0],
            [x3, y3, 67.5],
            [-2.0, 0.0, 90.0]
        ])
        
        # Generate interpolated poses
        poses = []
        for i in range(len(keyframes) - 1):
            p1 = keyframes[i]
            p2 = keyframes[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            # Linear interpolation in x, y, a
            # The task says: "linear interpolations of (x, y, a) in m ... equal steps"
            # This usually means we take m steps from p1 to p2.
            # Do we include p1 or p2? 
            # "Between consecutive rows it adds linear interpolations... in m equal steps"
            # If I have row A and row B, and m steps, do I add m points between them?
            # "adds linear interpolations ... in m equal steps"
            # Typically this means the segment is divided into m parts, creating m-1 intermediate points? 
            # Or m points total including endpoints?
            # "puts (0,0,0) before ... and (-2,0,90) after"
            # "Between consecutive rows it adds..."
            # If it adds m points, are the original rows included? Yes, "before/after" implies the returned rows are present.
            # So for each segment between returned row i and i+1, we generate m interpolated points.
            # If m=1, is it 1 point? Probably the midpoint? Or just 1 step?
            # "linear interpolations ... in m equal steps"
            # Let's assume it generates m points: t_k = k/m for k=1..m? Or k=0..m-1?
            # If m=1, one step. Usually "steps" implies intervals.
            # Let's look at the phrasing: "adds linear interpolations ... in m equal steps".
            # If I walk from A to B in 2 steps, I go A -> Mid -> B.
            # Does it add Mid? Yes.
            # Does it add A and B? No, they are already there.
            # So for m steps, we add m-1 intermediate points?
            # Or does "m equal steps" mean m points are added?
            # Let's assume m points are added between the two existing rows.
            # If m=1, 1 point added (midpoint).
            # If m=2, 2 points added (1/3, 2/3).
            # Let's verify with "m = max(1, ceil(|a2-a1|))".
            # If angle diff is 1, m=1. Add 1 point.
            # This seems reasonable.
            
            if m > 1:
                t_vals = np.linspace(0, 1, m+1)[1:-1] # Exclude endpoints
            else:
                t_vals = np.array([0.5])
            
            for t in t_vals:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                poses.append([x, y, a])
        
        # Add the keyframes themselves? 
        # The evaluator "puts (0,0,0) before your rows and (-2,0,90) after them".
        # And "Between consecutive rows it adds...".
        # So the full sequence is:
        # (0,0,0), [interps between 0 and row1], row1, [interps between row1 and row2], row2, ..., [-2,0,90]
        
        full_poses = [(0.0, 0.0, 0.0)]
        for i in range(len(keyframes) - 1):
            p1 = keyframes[i]
            p2 = keyframes[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            # The rows returned by us are keyframes[1:-1]
            # But in my loop above, I treated keyframes as the full set including start/end.
            # Let's restart the pose generation logic to match the evaluator exactly.
            
            pass # This block was just for thinking
            
        # Correct Pose Generation
        # Returned path is keyframes[1:-1]
        # Evaluator adds (0,0,0) at start, (-2,0,90) at end.
        # So the "rows" are:
        # R0 = (0,0,0)
        # R1...Rk = our returned points
        # R_{k+1} = (-2,0,90)
        
        # My `keyframes` array above IS the full sequence of "rows" (including start/end).
        # So I just need to insert interpolations between each consecutive pair in `keyframes`.
        
        all_poses = []
        # Add first point
        all_poses.append(keyframes[0])
        
        for i in range(len(keyframes) - 1):
            p1 = keyframes[i]
            p2 = keyframes[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            # Add m interpolated points?
            # "adds linear interpolations ... in m equal steps"
            # If m=1, 1 step. Does it add 1 point?
            # If I go from A to B in 1 step, I don't add any intermediate point.
            # But "m = max(1, ...)" suggests at least 1 step.
            # If it adds 0 points for m=1, then m=1 is same as no interpolation.
            # But the text says "adds ... in m equal steps".
            # Usually "N steps" implies N intervals, N-1 intermediate points.
            # However, if m=1, maybe it adds 1 point?
            # Let's assume it adds m points.
            # If m=1, add midpoint.
            # If m=2, add 1/3, 2/3.
            
            if m >= 1:
                # Generate m points
                # t_k = k / m for k=1..m?
                # If m=1, t=1/1=1? That's the endpoint.
                # So maybe t_k = k / (m+1)?
                # If m=1, t=1/2.
                # If m=2, t=1/3, 2/3.
                # This seems the most logical interpretation of "adding m points".
                t_vals = np.linspace(0, 1, m+1)[1:-1]
                for t in t_vals:
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    a = p1[2] + t * (p2[2] - p1[2])
                    all_poses.append([x, y, a])
            
            # Add the next keyframe (which is either an intermediate returned row or the final end)
            all_poses.append(p2)
            
        # Now all_poses contains all placements.
        # Check intersection of sofa points.
        
        valid_mask = np.ones(len(sofa_points), dtype=bool)
        
        for pose in all_poses:
            x, y, a_deg = pose
            a_rad = np.deg2rad(a_deg)
            
            # Rotate sofa points by -a_rad and translate by -x, -y
            # w' = R(-a) (w - c)
            # R(-a) = [[cos, sin], [-sin, cos]]
            
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = sofa_points[:, 0] - x
            dy = sofa_points[:, 1] - y
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            mask = inside_base(np.column_stack([u, v]))
            valid_mask &= mask
            
            if not np.any(valid_mask):
                break # Early exit if no points remain
        
        return np.sum(valid_mask) * 0.0001

    # Optimization
    # Initial guess based on current program
    # Current: x = -2t, y = 0.6 sin(pi t)
    # t=0.25 (22.5 deg): x=-0.5, y=0.6
    # t=0.5 (45 deg): x=-1.0, y=0.6
    # t=0.75 (67.5 deg): x=-1.5, y=0.6
    
    x0 = np.array([-0.5, 0.6, -1.0, 0.6, -1.5, 0.6])
    
    # Bounds
    # x should be roughly between -3 and 1
    # y should be roughly between -1 and 2
    bounds = [(-3, 1), (-2, 2), (-3, 1), (-2, 2), (-3, 1), (-2, 2)]
    
    # Use differential evolution for global search
    # This might be slow, so limit iterations/population
    result = differential_evolution(
        lambda p: -evaluate_path_params(p), 
        bounds, 
        maxiter=50, 
        popsize=10, 
        tol=0.01,
        seed=42
    )
    
    best_params = result.x
    x1, y1, x2, y2, x3, y3 = best_params
    
    path = np.array([
        [x1, y1, 22.5],
        [x2, y2, 45.0],
        [x3, y3, 67.5]
    ])
    
    return path
