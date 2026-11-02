import numpy as np
import scipy.optimize as opt
import time

N = 11

def get_hex_vertices(cx, cy, angle_rad):
    # Vertices of a unit regular hexagon centered at (cx, cy)
    # Side length 1
    # Angle 0 means vertex at (1, 0)
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + angle_rad
    xs = cx + np.cos(angles)
    ys = cy + np.sin(angles)
    return np.column_stack((xs, ys))

def point_in_polygon(p, poly):
    # Ray casting algorithm
    x, y = p
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segments_intersect(p1, p2, p3, p4):
    # Check if segment p1-p2 intersects p3-p4
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def check_overlap(v1, v2):
    # v1, v2 are (6,2) arrays
    # Check vertices
    for i in range(6):
        if point_in_polygon(v1[i], v2):
            return True
        if point_in_polygon(v2[i], v1):
            return True
    # Check edges
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def objective(params):
    # params: x1, y1, a1, x2, y2, a2, ...
    xs = params[0::3]
    ys = params[1::3]
    angles = params[2::3]
    
    # Calculate max support for outer hexagon (rot 0)
    # Outer normals
    normals = np.array([
        [np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))],
        [np.cos(np.deg2rad(120)), np.sin(np.deg2rad(120))],
        [np.cos(np.deg2rad(180)), np.sin(np.deg2rad(180))],
        [np.cos(np.deg2rad(240)), np.sin(np.deg2rad(240))],
        [np.cos(np.deg2rad(300)), np.sin(np.deg2rad(300))],
        [np.cos(np.deg2rad(0)), np.sin(np.deg2rad(0))]
    ])
    
    max_proj = 0.0
    overlap_penalty = 0.0
    
    vertices_list = []
    for i in range(N):
        v = get_hex_vertices(xs[i], ys[i], angles[i])
        vertices_list.append(v)
        for k in range(6):
            proj = np.max(np.dot(v, normals[k]))
            if proj > max_proj:
                max_proj = proj
                
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            if check_overlap(vertices_list[i], vertices_list[j]):
                overlap_penalty += 100.0 # Large penalty

    # S = 2/sqrt(3) * max_proj
    S = (2.0 / np.sqrt(3.0)) * max_proj
    return S + overlap_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: 7 central + 4 extra
    # Axial coords for 7 central
    axial = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    ]
    # Add 4 more, e.g., (2,0), (1,1), (0,2), (-1,2)
    axial += [
        (2,0), (1,1), (0,2), (-1,2)
    ]
    
    # Convert to Cartesian
    init_params = []
    for q, r in axial:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        init_params.append(x)
        init_params.append(y)
        init_params.append(0.0)
        
    init_params = np.array(init_params)
    
    best_result = None
    best_val = objective(init_params)
    
    # Local search
    # Use Nelder-Mead
    try:
        res = opt.minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if time.time() < deadline:
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x
            else:
                best_result = init_params
        else:
            best_result = init_params
            
    except Exception:
        best_result = init_params
        best_val = objective(best_result)
        
    # Extract results
    xs = best_result[0::3]
    ys = best_result[1::3]
    angles_rad = best_result[2::3]
    
    # Calculate final S
    # Re-evaluate to be sure
    max_proj = 0.0
    normals = np.array([
        [np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))],
        [np.cos(np.deg2rad(120)), np.sin(np.deg2rad(120))],
        [np.cos(np.deg2rad(180)), np.sin(np.deg2rad(180))],
        [np.cos(np.deg2rad(240)), np.sin(np.deg2rad(240))],
        [np.cos(np.deg2rad(300)), np.sin(np.deg2rad(300))],
        [np.cos(np.deg2rad(0)), np.sin(np.deg2rad(0))]
    ])
    for i in range(N):
        v = get_hex_vertices(xs[i], ys[i], angles_rad[i])
        for k in range(6):
            proj = np.max(np.dot(v, normals[k]))
            if proj > max_proj:
                max_proj = proj
    S = (2.0 / np.sqrt(3.0)) * max_proj
    
    inner = []
    for i in range(N):
        inner.append([xs[i], ys[i], np.rad2deg(angles_rad[i])])
        
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-START
# (The code above is the complete program, but we need to wrap it in the block format if required? 
# The prompt says "Keep the function names... Output the COMPLETE program in a single python code block.")
# I will output the whole file.
# EVOLVE-BLOCK-END
