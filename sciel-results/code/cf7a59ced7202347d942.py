"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def hex_face_normals(angle_deg):
    angle_rad = np.radians(angle_deg)
    normals = []
    for k in range(6):
        a = angle_rad + np.pi/6 + k * np.pi/3
        normals.append(np.array([np.cos(a), np.sin(a)]))
    return normals

def compute_outer_side(inner_pos, inner_ang, center, outer_ang_deg):
    """Compute outer hexagon side to contain all inner hexagons."""
    normals = hex_face_normals(outer_ang_deg)
    max_proj = 0
    for i in range(N):
        px, py = inner_pos[i]
        pa = inner_ang[i]
        verts = hex_vertices(px, py, pa, 1.0)
        for v in verts:
            rel = v - np.array(center)
            for n in normals:
                proj = np.dot(rel, n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj * 2.0 / np.sqrt(3)

def objective(params):
    positions = [(params[3*i], params[3*i+1]) for i in range(N)]
    angles = [params[3*i+2] for i in range(N)]
    cx = params[3*N]
    cy = params[3*N+1]
    outer_angle = params[3*N+2]
    return compute_outer_side(positions, angles, (cx, cy), outer_angle)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    d = np.sqrt(3)  # touching distance
    
    # Hexagonal cluster: center + 6 + 4 from second ring
    hex_pos = [(0.0, 0.0)]
    for k in range(6):
        a = k * np.pi / 3
        hex_pos.append((d * np.cos(a), d * np.sin(a)))
    for k in range(6):
        a = k * np.pi / 3 + np.pi/6
        hex_pos.append((2*d * np.cos(a), 2*d * np.sin(a)))
    hex_pos = hex_pos[:N]
    
    params0 = []
    for (x, y) in hex_pos:
        params0.extend([x, y, 0.0])
    params0.extend([0.0, 0.0, 0.0])
    
    best_params = np.array(params0, dtype=float)
    best_val = objective(best_params)
    
    # First optimization pass
    try:
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_val:
            best_params = res.x.copy()
            best_val = res.fun
    except:
        pass
    
    if time.time() > deadline:
        positions = [(best_params[3*i], best_params[3*i+1]) for i in range(N)]
        angles = [best_params[3*i+2] for i in range(N)]
        cx, cy = best_params[3*N], best_params[3*N+1]
        oa = best_params[3*N+2]
        side = compute_outer_side(positions, angles, (cx, cy), oa)
        inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
        return inner, (cx, cy), side, oa
    
    # Second pass with tighter tolerance
    try:
        res2 = minimize(objective, best_params, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
        if res2.fun < best_val:
            best_params = res2.x.copy()
            best_val = res2.fun
    except:
        pass
    
    positions = [(best_params[3*i], best_params[3*i+1]) for i in range(N)]
    angles = [best_params[3*i+2] for i in range(N)]
    cx, cy = best_params[3*N], best_params[3*N+1]
    oa = best_params[3*N+2]
    side = compute_outer_side(positions, angles, (cx, cy), oa)
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    return inner, (cx, cy), side, oa
# EVOLVE-BLOCK-END
