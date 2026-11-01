# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    import itertools
    
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = np.sqrt(3)
    
    # Axial to Cartesian
    def axial_to_xy(q, r):
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        return np.array([x, y])
        
    # Generate 19 points of a hexagon of radius 3
    R = 3
    points = []
    for q in range(-R, R+1):
        for r in range(max(-R, -q-R), min(R, -q+R)+1):
            points.append((q, r))
            
    coords = [axial_to_xy(q, r) for q, r in points]
    
    def min_hexagon_side_fast(pts_array):
        min_s = float('inf')
        best_angle = 0
        best_center = (0,0)
        
        cx, cy = np.mean(pts_array, axis=0)
        centered_pts = pts_array - np.array([cx, cy])
        
        # Coarse search
        for angle_deg in range(0, 60, 5):
            angle_rad = np.radians(angle_deg)
            normal_angles = angle_rad + np.radians(30) + np.radians(np.arange(0, 360, 60))
            cos_n = np.cos(normal_angles)
            sin_n = np.sin(normal_angles)
            projs = centered_pts @ np.column_stack([cos_n, sin_n]).T
            max_abs = np.max(np.abs(projs), axis=0)
            max_d = np.max(max_abs)
            s_req = 2 * max_d / sqrt3
            
            if s_req < min_s:
                min_s = s_req
                best_angle = angle_deg
                best_center = (cx, cy)
                
        # Fine search
        start_a = max(0, best_angle - 5)
        end_a = min(60, best_angle + 6)
        
        for angle_deg in range(start_a, end_a):
            angle_rad = np.radians(angle_deg)
            normal_angles = angle_rad + np.radians(30) + np.radians(np.arange(0, 360, 60))
            cos_n = np.cos(normal_angles)
            sin_n = np.sin(normal_angles)
            projs = centered_pts @ np.column_stack([cos_n, sin_n]).T
            max_abs = np.max(np.abs(projs), axis=0)
            max_d = np.max(max_abs)
            s_req = 2 * max_d / sqrt3
            
            if s_req < min_s:
                min_s = s_req
                best_angle = angle_deg
                best_center = (cx, cy)
                
        return min_s, best_angle, best_center

    best_side = float('inf')
    best_inner_coords = None
    best_outer_center = (0,0)
    best_outer_angle = 0
    
    # Iterate over combinations
    for combo in itertools.combinations(range(19), 12):
        if time.time() > deadline - 5:
            break
            
        idx = list(combo)
        pts = np.array([coords[i] for i in idx])
        
        min_x = np.min(pts[:, 0])
        max_x = np.max(pts[:, 0])
        min_y = np.min(pts[:, 1])
        max_y = np.max(pts[:, 1])
        
        w = max_x - min_x
        h = max_y - min_y
        
        lower_bound = max(w/2, h/sqrt3)
        
        if lower_bound >= best_side:
            continue
            
        s, ang, cen = min_hexagon_side_fast(pts)
        
        if s < best_side:
            best_side = s
            best_inner_coords = pts
            best_outer_center = cen
            best_outer_angle = ang

    inner = []
    if best_inner_coords is not None:
        for p in best_inner_coords:
            inner.append([p[0], p[1], 0.0])
        center = list(best_outer_center)
        side = best_side
        angle = best_outer_angle
    else:
        # Fallback
        inner = []
        for i in range(3):
            for j in range(4):
                x = j * 2.0 - 3.0
                y = i * 1.8 - 1.8
                inner.append([x, y, 0.0])
        center = [0, 0]
        side = 5.0
        angle = 0.0
        
    return inner, center, side, angle
# EVOLVE-BLOCK-END
