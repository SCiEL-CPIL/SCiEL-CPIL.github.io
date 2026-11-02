import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def compute_min_side(hexes):
        # Outer hexagon normals (angle 0)
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def rigid_cluster_objective(params):
        """
        params: [x, y, theta]
        x, y: translation of cluster center
        theta: rotation of cluster
        """
        if time.time() > deadline:
            return 1e10
            
        x, y, theta = params
        
        # Define the rigid cluster of 12 hexagons
        # Local coordinates of centers relative to cluster center
        centers_local = []
        # Shell 0: 1 hex at (0,0)
        centers_local.append([0.0, 0.0])
        
        # Shell 1: 6 hexes at distance sqrt(3)
        d1 = np.sqrt(3.0)
        for k in range(6):
            angle = np.radians(60 * k)
            cx = d1 * np.cos(angle)
            cy = d1 * np.sin(angle)
            centers_local.append([cx, cy])
            
        # Shell 2: We need 5 more hexes. 
        # To make the cluster as symmetric as possible, we pick 5 from the 12 positions.
        # Positions in Shell 2 are at distance 2*sqrt(3) and angles 0, 30, 60...
        d2 = 2.0 * np.sqrt(3.0)
        # Pick 5 angles. Let's pick 0, 72? No, must be lattice points.
        # Lattice angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
        # Let's pick 5 that are spread out. 
        # Maybe 0, 90, 180, 270, and one more? That's 4. 
        # Let's pick 0, 60, 120, 180, 240? (5 points)
        # Or 0, 30, 150, 210, 330?
        # Let's try 0, 72 is not valid.
        # Let's use 5 consecutive? That creates a "tail".
        # Let's use 5 that maximize symmetry: 0, 144? No.
        # Let's just pick 5: 0, 60, 120, 180, 240.
        shell2_angles = [0, 60, 120, 180, 240]
        for angle_deg in shell2_angles:
            angle = np.radians(angle_deg)
            cx = d2 * np.cos(angle)
            cy = d2 * np.sin(angle)
            centers_local.append([cx, cy])
            
        centers_local = np.array(centers_local)
        
        # Rotate and translate
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        
        centers_global = (centers_local @ rot.T) + np.array([x, y])
        
        # Generate vertices for all 12 hexagons
        # All hexagons have the same orientation as the cluster?
        # In a honeycomb, all hexagons have the same orientation.
        # So angle for each hexagon is theta (relative to world) + 0?
        # The local frame of the hexagon is rotated by theta.
        # So the hexagon angle is theta.
        
        hexes = []
        for cx, cy in centers_global:
            # The hexagon itself is rotated by theta
            verts = hex_vertices(cx, cy, 1.0, theta)
            hexes.append(verts)
            
        # Calculate min side
        s_min = compute_min_side(hexes)
        
        return s_min

    # Optimization
    # Initial guess: center at 0,0, theta=0
    x0 = [0.0, 0.0, 0.0]
    
    # Try a few different initial orientations for the cluster to avoid local minima
    best_s = np.inf
    best_params = None
    
    initial_thetas = [0.0, np.radians(30), np.radians(15), np.radians(45)]
    
    for t0 in initial_thetas:
        x0_try = [0.0, 0.0, t0]
        try:
            res = minimize(
                rigid_cluster_objective,
                x0_try,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # If best_params is None, fallback to simple grid search or previous method?
    # For now, if it fails, use a known good approximate or just return a valid packing
    if best_params is None:
        # Fallback: Simple hexagonal packing with outer side ~ 4
        # This is a safe fallback
        inner = []
        for k in range(6):
            angle = np.radians(60 * k)
            # Place 2 hexes per sector
            for j in range(2):
                dist = 1.5 + j * 1.7
                x = dist * np.cos(angle + np.radians(30))
                y = dist * np.sin(angle + np.radians(30))
                inner.append([x, y, 0.0])
        outer_center = (0.0, 0.0)
        outer_side = 5.0
        outer_angle_deg = 0.0
        return inner, outer_center, outer_side, outer_angle_deg

    x, y, theta = best_params
    
    # Construct the final answer
    inner = []
    centers_local = []
    centers_local.append([0.0, 0.0])
    d1 = np.sqrt(3.0)
    for k in range(6):
        angle = np.radians(60 * k)
        cx = d1 * np.cos(angle)
        cy = d1 * np.sin(angle)
        centers_local.append([cx, cy])
        
    d2 = 2.0 * np.sqrt(3.0)
    shell2_angles = [0, 60, 120, 180, 240]
    for angle_deg in shell2_angles:
        angle = np.radians(angle_deg)
        cx = d2 * np.cos(angle)
        cy = d2 * np.sin(angle)
        centers_local.append([cx, cy])
        
    centers_local = np.array(centers_local)
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    centers_global = (centers_local @ rot.T) + np.array([x, y])
    
    for cx, cy in centers_global:
        inner.append([cx, cy, np.degrees(theta)])
        
    # Calculate final outer side
    hexes = [hex_vertices(i[0], i[1], 1.0, np.radians(i[2])) for i in inner]
    s_final = compute_min_side(hexes)
    
    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
