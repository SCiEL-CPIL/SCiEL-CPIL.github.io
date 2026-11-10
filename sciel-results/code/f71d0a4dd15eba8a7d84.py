import numpy as np

# Pre-compute grid
u_grid = np.arange(-3, 1.5, 0.01)
v_grid = np.arange(-1.5, 1.5, 0.01)
U, V = np.meshgrid(u_grid, v_grid)
grid_points = np.column_stack((U.ravel(), V.ravel()))
N_POINTS = len(grid_points)

def compute_area(path_params):
    """
    path_params: list of tuples (x, y, a) for intermediate waypoints.
    Start (0,0,0) and End (-2,0,90) are added by evaluator, so we must include them in our simulation
    to match the evaluator's logic exactly.
    """
    # Construct full path with start and end
    # path_params are the rows we return.
    # Evaluator: Start, *path_params, End.
    # We simulate this.
    
    all_x = np.concatenate(([0.0], [p[0] for p in path_params], [-2.0]))
    all_y = np.concatenate(([0.0], [p[1] for p in path_params], [0.0]))
    all_a = np.concatenate(([0.0], [p[2] for p in path_params], [90.0]))
    
    # Interpolate to 1 degree steps (as per evaluator logic)
    # Evaluator: Between consecutive rows, m = max(1, ceil(|a2-a1|)).
    # Adds m-1 intermediate points.
    
    placements = []
    for i in range(len(all_x) - 1):
        x1, y1, a1 = all_x[i], all_y[i], all_a[i]
        x2, y2, a2 = all_x[i+1], all_y[i+1], all_a[i+1]
        m = max(1, int(np.ceil(np.abs(a2 - a1))))
        if m > 1:
            # m intervals, so m+1 points total. We already have start (i).
            # We need to add m-1 points between i and i+1.
            # Then add i+1.
            # t = 1/m, 2/m, ..., (m-1)/m
            for k in range(1, m):
                t = k / m
                xi = x1 + t * (x2 - x1)
                yi = y1 + t * (y2 - y1)
                ai = a1 + t * (a2 - a1)
                placements.append((xi, yi, ai))
        # Add the end point of this segment
        placements.append((x2, y2, a2))
        
    # Compute intersection
    # inside = True for all points initially
    inside = np.ones(N_POINTS, dtype=bool)
    
    for x, y, a in placements:
        rad = -np.radians(a)
        c = np.cos(rad)
        s = np.sin(rad)
        wx = grid_points[:, 0] - x
        wy = grid_points[:, 1] - y
        u = c * wx - s * wy
        v = s * wx + c * wy
        
        in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = in_rect1 | in_rect2
        
        inside &= in_base
        
        if not np.any(inside):
            return 0.0
            
    area = np.sum(inside) * 0.0001
    return area

def moving_sofa_2d() -> np.ndarray:
    # Define candidate paths
    # Each path is a list of (x, y, a) waypoints (excluding start/end which are fixed)
    
    candidates = []
    
    # Candidate 1: Linear path
    # Waypoints at a=30, a=60
    candidates.append([
        (-2.0 * (30/90), 0.0, 30.0),
        (-2.0 * (60/90), 0.0, 60.0)
    ])
    
    # Candidate 2: Bulge Up (y > 0)
    # Try to keep the "inner" part away from the outer corner?
    # L-shape inner corner is (0,0). Outer is (-3,-3) to (1,1) roughly.
    # If we move y up, we might clear the bottom part?
    candidates.append([
        (-2.0 * (30/90), 0.5, 30.0),
        (-2.0 * (60/90), 0.5, 60.0)
    ])
    
    # Candidate 3: Bulge Down (y < 0)
    candidates.append([
        (-2.0 * (30/90), -0.5, 30.0),
        (-2.0 * (60/90), -0.5, 60.0)
    ])
    
    # Candidate 4: Arc-like (Bulge Up then Down? or just Up)
    # Let's try a sine bulge
    candidates.append([
        (-2.0 * (30/90), 0.8 * np.sin(np.pi * 30/90), 30.0),
        (-2.0 * (60/90), 0.8 * np.sin(np.pi * 60/90), 60.0)
    ])
    
    # Candidate 5: Shifted Linear (Start moving x later?)
    # x = -2 * (t^2)
    candidates.append([
        (-2.0 * (30/90)**2, 0.0, 30.0),
        (-2.0 * (60/90)**2, 0.0, 60.0)
    ])
    
    # Candidate 6: Shifted Linear (Fast start)
    # x = -2 * (1 - (1-t)^2)
    candidates.append([
        (-2.0 * (1 - (1-30/90)**2), 0.0, 30.0),
        (-2.0 * (1 - (1-60/90)**2), 0.0, 60.0)
    ])
    
    # Candidate 7: More waypoints for finer control
    # a=20, 40, 70
    candidates.append([
        (-2.0 * (20/90), 0.0, 20.0),
        (-2.0 * (40/90), 0.3, 40.0),
        (-2.0 * (70/90), 0.0, 70.0)
    ])
    
    # Candidate 8: Asymmetric bulge
    candidates.append([
        (-2.0 * (30/90), 0.3, 30.0),
        (-2.0 * (60/90), -0.3, 60.0)
    ])
    
    best_path = None
    best_area = -1.0
    
    for path in candidates:
        area = compute_area(path)
        if area > best_area:
            best_area = area
            best_path = path
            
    # Return the best path as an array of shape (n, 3)
    return np.array(best_path)
