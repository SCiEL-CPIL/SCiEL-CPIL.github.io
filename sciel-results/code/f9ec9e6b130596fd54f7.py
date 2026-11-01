import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0

def hex_vertices(center, side=1.0, angle=0.0):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def objective(params):
    t_start = time.time()
    if t_start > DEADLINE:
        return 1e6

    centers = params[:N*2].reshape(N, 2)
    S = params[N*2]

    if S < 1.0:
        S = 1.0

    penalty = 0.0
    min_dist = np.sqrt(3)

    # Overlap
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                penalty += (min_dist - d) ** 2

    # Boundary
    apothem = S * np.sqrt(3) / 2.0
    normals = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    n_x = np.cos(normals)
    n_y = np.sin(normals)

    for i in range(N):
        verts = hex_vertices(centers[i], side=1.0, angle=0.0)
        px = verts[:, 0]
        py = verts[:, 1]
        projs = px[:, np.newaxis] * n_x[np.newaxis, :] + py[:, np.newaxis] * n_y[np.newaxis, :]
        max_proj = np.max(projs, axis=1)
        for mp in max_proj:
            if mp > apothem:
                penalty += (mp - apothem) ** 2

    return S + 100.0 * penalty

def construct():
    # Initial guess
    centers = []
    centers.append([0.0, 0.0])
    for k in range(6):
        angle = k * np.pi / 3
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append([x, y])
    for k in range(5):
        angle = k * np.pi / 3 + np.pi/6
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        centers.append([x, y])

    S_init = 4.0
    x0 = np.zeros(N*2 + 1)
    for i in range(N):
        x0[i*2] = centers[i][0]
        x0[i*2+1] = centers[i][1]
    x0[N*2] = S_init

    bounds = [(None, None) for _ in range(N*2)] + [(3.4, 6.0)]

    best_score = 1e9
    best_params = x0

    t0 = time.time()
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'fatol': 1e-8, 'xatol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass

    S_final = best_params[N*2]
    
    inner = []
    for i in range(N):
        x = best_params[i*2]
        y = best_params[i*2+1]
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S_final, 0.0
