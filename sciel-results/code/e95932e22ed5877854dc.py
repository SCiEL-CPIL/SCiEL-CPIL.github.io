import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _get_hex_vertices(cx, cy, angle):
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]) + angle
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _objective(params, normals, alpha=50.0, beta=50.0, lambda_overlap=10.0):
    X, Y = params[0], params[1]
    hex_params = params[2:].reshape(12, 3)
    
    centers = hex_params[:, :2] + np.array([X, Y])
    angles = hex_params[:, 2]
    
    # Bounding Box Term
    all_verts = np.empty((72, 2))
    for i in range(12):
        all_verts[i*6:(i+1)*6] = _get_hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        
    projs = all_verts @ normals.T
    max_projs = np.max(projs, axis=1, keepdims=True)
    m_k = np.log(np.sum(np.exp(alpha * (projs - max_projs)), axis=1)) / alpha + max_projs[:, 0]
    max_m = np.max(m_k, keepdims=True)
    F_box = np.log(np.sum(np.exp(beta * (m_k - max_m)))) / beta + max_m
    
    # Overlap Penalty
    # Distance between centers
    # centers is 12x2
    # Compute pairwise distances
    # Use broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(12, dtype=bool)
    dists_masked = dists[mask]
    
    # Penalty for dist < sqrt(3)
    min_dist = np.sqrt(3.0)
    violations = np.maximum(0, min_dist - dists_masked)
    F_overlap = np.sum(violations ** 2)
    
    return F_box[0] + lambda_overlap * F_overlap

def _init_honeycomb():
    sqrt3 = np.sqrt(3)
    params = np.zeros(2 + 12*3)
    params[0] = 0.0
    params[1] = 0.0
    
    idx = 2
    params[idx:idx+3] = [0.0, 0.0, 0.0]
    idx += 3
    
    for k in range(6):
        angle = k * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        params[idx:idx+3] = [x, y, 0.0]
        idx += 3
        
    for k in range(5):
        angle = k * np.pi / 3
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        params[idx:idx+3] = [x, y, 0.0]
        idx += 3
        
    return params

def _init_grid(rows, cols):
    params = np.zeros(2 + 12*3)
    params[0] = 0.0
    params[1] = 0.0
    
    sqrt3 = np.sqrt(3)
    idx = 2
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= 12: break
            x = c * sqrt3 + (r % 2) * sqrt3/2
            y = r * 1.5
            params[idx:idx+3] = [x, y, 0.0]
            idx += 3
            count += 1
    xs = params[2::3]
    ys = params[3::3]
    cx = (np.max(xs) + np.min(xs)) / 2
    cy = (np.max(ys) + np.min(ys)) / 2
    params[2::3] -= cx
    params[3::3] -= cy
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n_angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.column_stack((np.cos(n_angles), np.sin(n_angles)))
    
    best_score = float('inf')
    best_params = None
    
    inits = [
        _init_honeycomb(),
        _init_grid(2, 6),
        _init_grid(3, 4),
        _init_grid(4, 3)
    ]
    
    # Add random
    rand_params = np.random.uniform(-5, 5, size=2 + 12*3)
    rand_params[2::3] = np.random.uniform(-3, 3, 12)
    rand_params[3::3] = np.random.uniform(-3, 3, 12)
    rand_params[4::3] = np.random.uniform(0, 2*np.pi, 12)
    inits.append(rand_params)
    
    for init in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective, 
                init, 
                args=(normals,), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        best_params = _init_honeycomb()
        best_score = _objective(best_params, normals)
        
    # The objective includes overlap penalty.
    # We need the true bounding box size.
    # Recalculate F_box without penalty
    X, Y = best_params[0], best_params[1]
    hex_params = best_params[2:].reshape(12, 3)
    centers = hex_params[:, :2] + np.array([X, Y])
    angles = hex_params[:, 2]
    
    all_verts = np.empty((72, 2))
    for i in range(12):
        all_verts[i*6:(i+1)*6] = _get_hex_vertices(centers[i, 0], centers[i, 1], angles[i])
    projs = all_verts @ normals.T
    H = np.max(projs)
    
    side_length = (2.0 / np.sqrt(3.0)) * H
    
    inner = []
    for i in range(12):
        x = X + hex_params[i, 0]
        y = Y + hex_params[i, 1]
        angle_deg = np.degrees(hex_params[i, 2]) % 360.0
        inner.append([x, y, angle_deg])
        
    return inner, (0.0, 0.0), side_length, 0.0

# EVOLVE-BLOCK-START
# (Code above is the implementation)
# EVOLVE-BLOCK-END
