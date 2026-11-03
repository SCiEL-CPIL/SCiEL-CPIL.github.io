import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 145.0

def hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = np.deg2rad(angle_deg + 30 + np.arange(6) * 60)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def outer_hex_normals():
    angles = np.deg2rad(30 + np.arange(6) * 60)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def objective(params, normals):
    positions = params.reshape(N, 3)
    max_val = 0.0
    for i in range(N):
        cx, cy, ang = positions[i]
        verts = hex_vertices(cx, cy, ang)
        dots = np.abs(verts @ normals.T)
        local_max = np.max(dots)
        if local_max > max_val:
            max_val = local_max
    return max_val / (np.sqrt(3) / 2)

def construct():
    start_time = time.time()
    normals = outer_hex_normals()
    sqrt3 = np.sqrt(3)
    
    # Initial guess: honeycomb-like
    init = np.zeros((N, 3))
    init[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        ang = k * 60
        init[k+1] = [sqrt3 * np.cos(np.deg2rad(ang)), sqrt3 * np.sin(np.deg2rad(ang)), 0.0]
    for k in range(4):
        ang = 30 + k * 60
        init[k+7] = [2*sqrt3 * np.cos(np.deg2rad(ang)), 2*sqrt3 * np.sin(np.deg2rad(ang)), 0.0]
    
    best_params = init.copy()
    best_val = objective(init, normals)
    
    np.random.seed(42)
    
    # Multiple restarts
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        if r == 0:
            params0 = init.copy()
        else:
            params0 = init + np.random.randn(N, 3) * 0.3
            params0[:, 2] *= 20
        
        try:
            result = minimize(objective, params0, args=(normals,), method='Powell',
                             options={'maxiter': 2000, 'xtol': 1e-7, 'ftol': 1e-9})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    best_positions = best_params.reshape(N, 3)
    inner = [[float(x), float(y), float(a)] for x, y, a in best_positions]
    outer_side = float(best_val)
    
    return inner, (0.0, 0.0), outer_side, 0.0
