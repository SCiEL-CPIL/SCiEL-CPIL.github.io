import numpy as np
import time
from scipy.optimize import minimize

N = 12

def get_hex_vertices(center, side, angle_deg):
    """
    Returns the 6 vertices of a regular hexagon.
    center: (x, y)
    side: length
    angle_deg: rotation in degrees
    """
    cx, cy = center
    angle_rad = np.deg2rad(angle_deg)
    # Vertices of a unit hexagon (pointy-top) are at angles 30, 90, 150, 210, 270, 330?
    # Let's use flat-top for standard tiling? 
    # Pointy-top vertices: (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
    # Let's stick to the definition used in the analysis: Pointy-Top.
    # Vertices relative to center for side 1:
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    # Wait, pointy top has a vertex at 90 deg (top).
    # So angles are 90, 30, -30, -90, -150, 150.
    # Let's just generate them and rotate.
    base_angles = np.deg2rad([90, 30, -30, -90, -150, 150])
    vertices = np.zeros((6, 2))
    for i, ang in enumerate(base_angles):
        vertices[i] = [side * np.cos(ang), side * np.sin(ang)]
    
    # Rotate
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])
    vertices = vertices @ rot_matrix.T
    vertices += np.array([cx, cy])
    return vertices

def calc_min_side(inner, outer_center, outer_angle_deg):
    """
    Calculate the minimum side length of the outer hexagon required to contain all inner hexagons.
    inner: list of [x, y, angle_deg]
    outer_center: (x, y)
    outer_angle_deg: rotation of outer hexagon
    """
    ocx, ocy = outer_center
    outer_vertices_normals = []
    # The outer hexagon is defined by 6 half-planes.
    # Normal vectors point outward.
    # For a pointy-top hexagon (vertex at 90), the normals are at 0, 60, 120, 180, 240, 300?
    # Let's check.
    # Vertex at 90. Edge from 30 to 90. Midpoint angle 60. Normal is 60.
    # Edge from -30 to 30. Midpoint 0. Normal 0.
    # So normals are at 0, 60, 120, 180, 240, 300.
    normal_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    
    # Rotate normals by outer_angle
    c = np.cos(np.deg2rad(outer_angle_deg))
    s = np.sin(np.deg2rad(outer_angle_deg))
    rot_matrix = np.array([[c, -s], [s, c]])
    
    normals = []
    for ang in normal_angles:
        n = np.array([np.cos(ang), np.sin(ang)])
        n_rot = rot_matrix @ n
        normals.append(n_rot)
    
    max_required_side = 0.0
    
    for ix, iy, iang in inner:
        # Get vertices of this inner hexagon
        verts = get_hex_vertices((ix, iy), 1.0, iang)
        for v in verts:
            # Check distance from outer_center to this vertex in direction of each normal
            vec = v - np.array(outer_center)
            for n in normals:
                proj = np.dot(vec, n)
                # The boundary is at distance apothem = S * sqrt(3)/2
                # So we need proj <= S * sqrt(3)/2
                # S >= proj / (sqrt(3)/2)
                if proj > 0: # Only consider projections in the direction of the normal
                    req_side = proj / (np.sqrt(3) / 2)
                    if req_side > max_required_side:
                        max_required_side = req_side
                        
    return max_required_side

def construct():
    start_time = time.time()
    deadline = 100.0 # Safety margin
    
    # Initial Configurations
    configs = []
    
    # Config 1: 2x6 Strip (Pointy Top)
    # Centers for 2x6 grid
    # Row 0: y=0, x = 0, sqrt(3), ...
    # Row 1: y=1.5, x = sqrt(3)/2, ...
    sqrt3 = np.sqrt(3)
    inner_2x6 = []
    for j in range(2):
        for i in range(6):
            x = i * sqrt3 + (j * sqrt3 / 2)
            y = j * 1.5
            # Center the cluster
            # Approx center x: (0 + 5*sqrt3 + sqrt3/2 + ...)/12
            # Let's just center by subtracting mean
            inner_2x6.append([x, y, 0.0])
    
    # Center inner_2x6
    xs = [p[0] for p in inner_2x6]
    ys = [p[1] for p in inner_2x6]
    cx = np.mean(xs)
    cy = np.mean(ys)
    for p in inner_2x6:
        p[0] -= cx
        p[1] -= cy
        
    configs.append((inner_2x6, (0.0, 0.0), 0.0))
    
    # Config 2: 3x4 Strip
    inner_3x4 = []
    for j in range(3):
        for i in range(4):
            x = i * sqrt3 + (j * sqrt3 / 2)
            y = j * 1.5
            inner_3x4.append([x, y, 0.0])
            
    xs = [p[0] for p in inner_3x4]
    ys = [p[1] for p in inner_3x4]
    cx = np.mean(xs)
    cy = np.mean(ys)
    for p in inner_3x4:
        p[0] -= cx
        p[1] -= cy
        
    configs.append((inner_3x4, (0.0, 0.0), 0.0))

    best_score = float('inf')
    best_inner = configs[0][0]
    best_center = configs[0][1]
    best_angle = configs[0][2]
    best_side = calc_min_side(best_inner, best_center, best_angle)
    
    for idx, (inner, center, angle) in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Optimize outer angle and center
        def objective(params):
            ocx, ocy, oang = params
            try:
                side = calc_min_side(inner, (ocx, ocy), oang)
                return side
            except:
                return 100.0
                
        x0 = [center[0], center[1], angle]
        
        # Simple Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.success or res.fun < best_score:
                ocx, ocy, oang = res.x
                side = res.fun
                if side < best_score:
                    best_score = side
                    best_inner = [p[:] for p in inner]
                    best_center = (ocx, ocy)
                    best_angle = oang
        except Exception as e:
            pass
            
        # Also try just the initial config if optimization failed or was slow
        side_init = calc_min_side(inner, center, angle)
        if side_init < best_score:
            best_score = side_init
            best_inner = [p[:] for p in inner]
            best_center = center
            best_angle = angle

    # Final check
    if time.time() > deadline - 5:
         # Ensure we return a valid result
         pass

    # Return best found
    return best_inner, best_center, best_score, best_angle

# EVOLVE-BLOCK-END
