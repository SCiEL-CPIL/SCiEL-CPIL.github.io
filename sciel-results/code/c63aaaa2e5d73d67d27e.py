import numpy as np
import time
from scipy.optimize import minimize

N = 32
TIME_LIMIT = 120.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # 1. Initialization: Hexagonal lattice
    points = []
    for row in range(10):
        for col in range(10):
            x = col + (0.5 if row % 2 else 0)
            y = row * np.sqrt(3) / 2
            points.append((x, y))
    
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    init_pts = pts[idx].copy()
    
    # Scale to fit
    min_x, min_y = init_pts.min(axis=0)
    max_x, max_y = init_pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    scale = 0.9 / max(width, height)
    
    init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
    init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
    
    # Center
    current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
    offset_x = (1.0 - current_w[0]) / 2
    offset_y = (1.0 - current_w[1]) / 2
    init_pts[:, 0] += offset_x
    init_pts[:, 1] += offset_y
    
    # Initial radii
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(init_pts[i] - init_pts[j])
            if d < min_dist:
                min_dist = d
    init_r = min_dist / 2 * 0.9
    
    # Variables
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = init_pts[i, 0]
        x0[3*i+1] = init_pts[i, 1]
        x0[3*i+2] = init_r
        
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))
        
    def objective(vars):
        X = vars[0::3]
        Y = vars[1::3]
        R = vars[2::3]
        
        obj = -np.sum(R)
        pen = 0.0
        
        for i in range(N):
            for j in range(i+1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    pen += overlap**2
        
        for i in range(N):
            if X[i] - R[i] < 0: pen += (X[i] - R[i])**2
            if 1.0 - X[i] - R[i] < 0: pen += (1.0 - X[i] - R[i])**2
            if Y[i] - R[i] < 0: pen += (Y[i] - R[i])**2
            if 1.0 - Y[i] - R[i] < 0: pen += (1.0 - Y[i] - R[i])**2
            
        weight = 100.0
        return obj + weight * pen

    def objective_grad(vars):
        X = vars[0::3]
        Y = vars[1::3]
        R = vars[2::3]
        
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        weight = 100.0
        
        for i in range(N):
            for j in range(i+1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-10:
                    continue
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    factor = 2.0 * overlap
                    grad[3*i]   += weight * factor * (-dx/d)
                    grad[3*i+1] += weight * factor * (-dy/d)
                    grad[3*j]   += weight * factor * (dx/d)
                    grad[3*j+1] += weight * factor * (dy/d)
                    grad[3*i+2] += weight * factor
                    grad[3*j+2] += weight * factor

        for i in range(N):
            val = X[i] - R[i]
            if val < 0:
                grad[3*i]   += weight * 2.0 * val
                grad[3*i+2] += weight * -2.0 * val
            
            val = 1.0 - X[i] - R[i]
            if val < 0:
                grad[3*i]   += weight * -2.0 * val
                grad[3*i+2] += weight * -2.0 * val
            
            val = Y[i] - R[i]
            if val < 0:
                grad[3*i+1] += weight * 2.0 * val
                grad[3*i+2] += weight * -2.0 * val
            
            val = 1.0 - Y[i] - R[i]
            if val < 0:
                grad[3*i+1] += weight * -2.0 * val
                grad[3*i+2] += weight * -2.0 * val

        return grad

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT * 0.7:
        return np.column_stack((init_pts[:, 0], init_pts[:, 1], np.full(N, init_r)))

    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=objective_grad, bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success:
            x_opt = res.x
        else:
            x_opt = x0
    except Exception:
        x_opt = x0

    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT * 0.9:
        pos = np.column_stack((x_opt[0::3], x_opt[1::3]))
        rad = x_opt[2::3].copy()
        # Quick polish
        for i in range(N):
             min_dist_others = np.inf
             for j in range(N):
                 if i == j: continue
                 d = np.linalg.norm(pos[i] - pos[j])
                 min_dist_others = min(min_dist_others, d - rad[j])
             min_dist_bound = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
             rad[i] = max(1e-6, min(min_dist_others, min_dist_bound) - 1e-5)
        return np.column_stack((pos[:, 0], pos[:, 1], rad))

    pos = x_opt[0::3].copy()
    rad = x_opt[2::3].copy()
    # Wait, pos extraction needs Y too
    X = x_opt[0::3].copy()
    Y = x_opt[1::3].copy()
    pos = np.column_stack((X, Y))
    
    # Polish
    for _ in range(5):
        for i in range(N):
            min_dist_others = np.inf
            for j in range(N):
                if i == j: continue
                d = np.linalg.norm(pos[i] - pos[j])
                min_dist_others = min(min_dist_others, d - rad[j])
            min_dist_bound = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            rad[i] = max(1e-6, min(min_dist_others, min_dist_bound) - 1e-5)
            
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = pos[i, 0]
        result[i, 1] = pos[i, 1]
        result[i, 2] = rad[i]
        
    return result
# EVOLVE-BLOCK-END
