import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_normals(angle_deg):
    a = np.radians(angle_deg)
    return np.array([[np.cos(a + k * np.pi / 3), np.sin(a + k * np.pi / 3)] for k in range(6)])

def _outer_side_length(points, center, angle_deg):
    normals = _hex_normals(angle_deg)
    diffs = points - center
    projections = diffs @ normals.T
    max_proj = np.max(projections)
    return 2.0 * max_proj / np.sqrt(3.0)

def construct():
    deadline = time.time() + 140.0
    sqrt3 = np.sqrt(3.0)
    
    # Honeycomb initial arrangement (pointy-top)
    positions = []
    row = 0
    while len(positions) < N:
        x_offset = 0.0 if row % 2 == 0 else sqrt3 / 2.0
        remaining = N - len(positions)
        for i in range(remaining):
            x = x_offset + i * sqrt3
            y = row * 1.5
            positions.append([x, y])
        row += 1
    
    pts = np.array(positions[:N], dtype=float)
    center = np.mean(pts, axis=0)
    outer_angle = 0.0
    
    def objective(params):
        n = N
        pos = params[:2 * n].reshape(n, 2)
        cx = params[2 * n]
        cy = params[2 * n + 1]
        oa = params[2 * n + 2]
        
        if n > 1:
            diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            penalty = max(0, sqrt3 - min_dist) ** 2 * 1000
        else:
            penalty = 0.0
        
        sl = _outer_side_length(pos, np.array([cx, cy]), oa)
        return sl + penalty
    
    init_params = np.concatenate([pts.flatten(), [center[0], center[1], outer_angle]])
    
    best_sl = _outer_side_length(pts, center, outer_angle)
    best_params = init_params.copy()
    
    try:
        result = minimize(objective, init_params, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < best_sl:
            best_sl = result.fun
            best_params = result.x
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        pos = pts
        cx, cy = center
        oa = outer_angle
    else:
        pos = best_params[:2 * N].reshape(N, 2)
        cx = best_params[2 * N]
        cy = best_params[2 * N + 1]
        oa = best_params[2 * N + 2]
    
    actual_sl = _outer_side_length(pos, np.array([cx, cy]), oa)
    
    inner = [[pos[i][0], pos[i][1], 0.0] for i in range(N)]
    return inner, (cx, cy), actual_sl, oa
