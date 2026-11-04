import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    
    # 1. Generate 12 hexagons in a hex grid (spiral order)
    # Axial coordinates (q, r)
    hexes_axial = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,0), (2,-1), (2,-2), (1,-2), (0,-2)
    ]
    
    centers = []
    for q, r in hexes_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        centers.append([x, y])
    centers = np.array(centers)
    
    # 2. Generate vertices of a unit flat-top hexagon
    # Vertices at 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    v_base = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # All vertices for all 12 hexagons (at theta=0)
    # Shape: (12, 6, 2)
    vertices = centers[:, np.newaxis, :] + v_base[np.newaxis, :, :]
    # Flatten to (72, 2)
    all_vertices = vertices.reshape(-1, 2)
    
    # 3. Normals for the container (flat-top)
    # Normals at 90, 150, 210, 270, 330, 30 degrees
    n_angles = np.deg2rad(np.array([90, 150, 210, 270, 330, 30]))
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    def calc_side(theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        rot = np.array([[c, -s], [s, c]])
        
        # Rotate normals by -theta
        n_rot = normals @ rot.T
        
        # Project all vertices onto rotated normals
        proj = all_vertices @ n_rot.T
        max_proj = np.max(proj, axis=0)
        
        # S = max( max_proj / (sqrt(3)/2) )
        s_val = np.max(max_proj) / (np.sqrt(3) / 2)
        return s_val
    
    # 4. Golden Section Search
    phi = (np.sqrt(5) - 1) / 2
    a, b = 0.0, 60.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = calc_side(c)
    fd = calc_side(d)
    
    best_theta = (a + b) / 2
    best_side = fc if fc < fd else fd
    
    max_iter = 200
    for i in range(max_iter):
        if time.time() - start_time > 110: break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = calc_side(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = calc_side(d)
            
        current_best = fc if fc < fd else fd
        if current_best < best_side:
            best_side = current_best
            best_theta = c if fc < fd else d
    
    best_theta_deg = best_theta
    best_side_val = best_side
    
    # 5. Construct output
    theta_rad = np.deg2rad(best_theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    rot = np.array([[c, -s], [s, c]])
    
    # Rotate centers
    rotated_centers = centers @ rot.T
    
    inner = []
    for i in range(N):
        x, y = rotated_centers[i]
        inner.append([x, y, best_theta_deg])
        
    return inner, (0.0, 0.0), best_side_val, 0.0
