import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(center, side, angle_deg):
    cx, cy = center
    angle = np.deg2rad(angle_deg)
    angles = angle + np.deg2rad(np.arange(6) * 60)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def outer_hexagon_side_length(inner_centers, inner_angles, outer_center, outer_angle):
    outer_angle_rad = np.deg2rad(outer_angle)
    normal_angles = outer_angle_rad + np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_proj = 0.0
    for i in range(N):
        verts = hexagon_vertices(inner_centers[i], 1.0, inner_angles[i])
        for v in verts:
            diff = v - outer_center
            projs = np.dot(normals, diff)
            current_max = np.max(np.abs(projs))
            if current_max > max_proj:
                max_proj = current_max
                
    apothem = max_proj
    side_length = 2.0 * apothem / np.sqrt(3.0)
    return side_length

def penalty_term(inner_centers):
    pen = 0.0
    min_dist = np.sqrt(3.0)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(inner_centers[i] - inner_centers[j])
            if d < min_dist:
                pen += (min_dist - d)**2
    return pen * 100.0

def objective(params):
    inner_centers = np.zeros((N, 2))
    for i in range(N):
        inner_centers[i][0] = params[2*i]
        inner_centers[i][1] = params[2*i+1]
    inner_angles = params[22:33]
    outer_center = np.array([params[33], params[34]])
    outer_angle = params[35]
    
    if not np.all(np.isfinite(params)):
        return 1e10
        
    side_len = outer_hexagon_side_length(inner_centers, inner_angles, outer_center, outer_angle)
    pen = penalty_term(inner_centers)
    
    return side_len + pen

def get_valid_side(params):
    inner_centers = np.zeros((N, 2))
    for i in range(N):
        inner_centers[i][0] = params[2*i]
        inner_centers[i][1] = params[2*i+1]
    inner_angles = params[22:33]
    outer_center = np.array([params[33], params[34]])
    outer_angle = params[35]
    
    pen = penalty_term(inner_centers)
    if pen > 1e-6:
        return None
    return outer_hexagon_side_length(inner_centers, inner_angles, outer_center, outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_valid_side = 1e10
    best_valid_params = None
    
    inits = []
    
    # Init 1: 3x3 grid + 2
    centers_3x3 = [
        [-np.sqrt(3), -1.5], [0, -1.5], [np.sqrt(3), -1.5],
        [-np.sqrt(3), 0], [0, 0], [np.sqrt(3), 0],
        [-np.sqrt(3), 1.5], [0, 1.5], [np.sqrt(3), 1.5],
        [0, 3.0], [0, -3.0]
    ]
    angles_11 = np.zeros(11)
    outer_c = [0, 0]
    outer_a = 0
    p1 = np.array([c for coord in centers_3x3 for c in coord] + list(angles_11) + outer_c + [outer_a])
    inits.append(p1)
    
    # Init 2: 7 cluster + 4
    centers_7 = [[0,0]]
    for k in range(6):
        ang = np.deg2rad(k * 60)
        x = np.sqrt(3) * np.cos(ang)
        y = np.sqrt(3) * np.sin(ang)
        centers_7.append([x, y])
    centers_7.extend([[3, 0], [-3, 0], [0, 3], [0, -3]])
    p2 = np.array([c for coord in centers_7 for c in coord] + list(angles_11) + outer_c + [outer_a])
    inits.append(p2)
    
    # Init 3: Random
    np.random.seed(42)
    p3 = p1 + np.random.normal(0, 0.5, size=len(p1))
    inits.append(p3)
    
    for p0 in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, p0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6})
            side = get_valid_side(res.x)
            if side is not None and side < best_valid_side:
                best_valid_side = side
                best_valid_params = res.x
        except:
            pass
            
    # More random restarts if time permits
    while time.time() < deadline - 10:
        p_rand = np.random.normal(0, 2.0, size=36)
        p_rand[22:33] = np.random.uniform(-15, 15, 11)
        try:
            res = minimize(objective, p_rand, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-5})
            side = get_valid_side(res.x)
            if side is not None and side < best_valid_side:
                best_valid_side = side
                best_valid_params = res.x
        except:
            pass

    if best_valid_params is None:
        # Fallback
        best_valid_params = p1
        best_valid_side = outer_hexagon_side_length(
            np.array([[p1[2*i], p1[2*i+1]] for i in range(N)]),
            p1[22:33],
            np.array([p1[33], p1[34]]),
            p1[35]
        )

    inner = []
    for i in range(N):
        inner.append([best_valid_params[2*i], best_valid_params[2*i+1], best_valid_params[22+i]])
        
    cx = best_valid_params[33]
    cy = best_valid_params[34]
    side = best_valid_side
    angle = best_valid_params[35]
    
    return inner, (cx, cy), side, angle

