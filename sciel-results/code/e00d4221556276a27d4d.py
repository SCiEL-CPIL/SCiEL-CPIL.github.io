import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    angles = np.deg2rad(np.arange(0, 360, 60))
    hex_verts = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    
    centers = np.zeros((N, 2))
    centers[0] = [0.0, 0.0]
    dist1 = np.sqrt(3)
    for i in range(1, 7):
        ang = np.deg2rad(60 * (i-1))
        centers[i] = [dist1 * np.cos(ang), dist1 * np.sin(ang)]
    dist2 = 3.0
    angles2 = np.deg2rad([30, 90, 150, 210, 270])
    for i in range(7, 12):
        centers[7 + (i-7)] = [dist2 * np.cos(angles2[i-7]), dist2 * np.sin(angles2[i-7])]
        
    S_init = 4.0
    x0 = np.concatenate([centers.flatten(), [S_init]])
    
    def objective(vars):
        return vars[-1]
        
    def constraints(vars):
        S = vars[-1]
        centers = vars[:-1].reshape(N, 2)
        cons = []
        
        # Non-overlap
        # Compute all pair distances
        # Use broadcasting
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # This creates a 12x12 matrix.
        # We need upper triangle.
        # d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        # triu_indices = np.triu_indices(N, k=1)
        # cons.extend(d[triu_indices] - np.sqrt(3))
        
        d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        triu_indices = np.triu_indices(N, k=1)
        cons = list(d[triu_indices] - np.sqrt(3))
        
        # Containment
        # For each center, add vertices
        # vs = centers[:, np.newaxis, :] + hex_verts[np.newaxis, :, :] # (N, 6, 2)
        # We need to check 6 inequalities for each vertex.
        # Let's unroll for speed or use matrix ops.
        
        for i in range(N):
            c = centers[i]
            vs = c + hex_verts
            xs = vs[:, 0]
            ys = vs[:, 1]
            
            # 1. sqrt(3)*S - (sqrt(3)*x + y) >= 0
            cons.append(np.sqrt(3)*S - (np.sqrt(3)*xs + ys))
            # 2. sqrt(3)/2*S - y >= 0
            cons.append(np.sqrt(3)/2*S - ys)
            # 3. sqrt(3)*S - (-sqrt(3)*x + y) >= 0
            cons.append(np.sqrt(3)*S - (-np.sqrt(3)*xs + ys))
            # 4. sqrt(3)*S - (-sqrt(3)*x - y) >= 0
            cons.append(np.sqrt(3)*S - (-np.sqrt(3)*xs - ys))
            # 5. y + sqrt(3)/2*S >= 0
            cons.append(ys + np.sqrt(3)/2*S)
            # 6. sqrt(3)*S - (sqrt(3)*x - y) >= 0
            cons.append(np.sqrt(3)*S - (np.sqrt(3)*xs - ys))
            
        return cons
        
    bounds = [(-10, 10)] * (2*N) + [(0.1, 20)]
    
    best_result = None
    best_score = float('inf')
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': constraints}], 
                       options={'maxiter': 1000, 'ftol': 1e-10})
        if res.fun < best_score:
            best_result = res
            best_score = res.fun
    except Exception:
        pass
        
    if time.time() > deadline:
        if best_result is None:
            return [[float(c[0]), float(c[1]), 0.0] for c in centers], (0.0, 0.0), S_init, 0.0
        else:
            final_vars = best_result.x
            final_centers = final_vars[:-1].reshape(N, 2)
            final_S = final_vars[-1]
            return [[float(c[0]), float(c[1]), 0.0] for c in final_centers], (0.0, 0.0), float(final_S), 0.0

    if best_result is not None:
        final_vars = best_result.x
        final_centers = final_vars[:-1].reshape(N, 2)
        final_S = final_vars[-1]
        return [[float(c[0]), float(c[1]), 0.0] for c in final_centers], (0.0, 0.0), float(final_S), 0.0
    else:
        return [[float(c[0]), float(c[1]), 0.0] for c in centers], (0.0, 0.0), S_init, 0.0
