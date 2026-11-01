import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 145.0

def hex_vertices():
    angles = np.arange(6) * np.pi / 3
    return np.column_stack((np.cos(angles), np.sin(angles)))

def container_normals():
    angles = np.arange(6) * np.pi / 3 + np.pi / 6
    return np.column_stack((np.cos(angles), np.sin(angles)))

NORMALS = container_normals()
VERTS = hex_vertices()

def compute_S(positions):
    max_proj = 0.0
    for i in range(len(positions)):
        p = positions[i]
        verts = VERTS + p
        projs = verts @ NORMALS.T
        max_proj = max(max_proj, np.max(projs))
    return 2.0 * max_proj / np.sqrt(3)

def overlap_penalty(positions, tol=1e-4):
    penalty = 0.0
    min_dist = np.sqrt(3) - tol
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                penalty += (min_dist - d) ** 2
    return penalty

def objective(vars_):
    positions = vars_.reshape(N, 2)
    S = compute_S(positions)
    pen = overlap_penalty(positions)
    return S + 100.0 * pen

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Initialize
    pos = np.zeros((N, 2))
    r1 = np.sqrt(3)
    for k in range(6):
        ang = k * np.pi / 3
        pos[1 + k] = [r1 * np.cos(ang), r1 * np.sin(ang)]
    r2 = 2 * np.sqrt(3)
    ring2_angles = [0, np.pi/3, 2*np.pi/3, np.pi]
    for k in range(4):
        ang = ring2_angles[k]
        pos[7 + k] = [r2 * np.cos(ang), r2 * np.sin(ang)]
        
    initial_vars = pos.flatten()
    best_vars = initial_vars
    best_score = objective(initial_vars)
    
    for restart in range(10):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        if restart == 0:
            x0 = initial_vars
        else:
            x0 = initial_vars + np.random.normal(0, 0.5, size=initial_vars.shape)
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_vars = res.x
        except Exception:
            pass

    positions = best_vars.reshape(N, 2)
    S = compute_S(positions)
    
    # Verify validity
    pen = overlap_penalty(positions, tol=1e-6)
    if pen > 1e-5:
        # Revert to initial if invalid
        positions = initial_vars.reshape(N, 2)
        S = compute_S(positions)

    inner = []
    for i in range(N):
        inner.append([float(positions[i][0]), float(positions[i][1]), 0.0])
        
    return inner, (0.0, 0.0), float(S), 0.0

# EVOLVE-BLOCK-START
# (Function is defined above)
# EVOLVE-BLOCK-END
