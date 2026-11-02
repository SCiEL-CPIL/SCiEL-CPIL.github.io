import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    # Lattice vectors
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate 19 points (hex cluster radius 2)
    coords = []
    for r in range(-2, 3):
        for q in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                p = q*u + r*v
                coords.append(p)
                
    coords = np.array(coords)
    dists = np.linalg.norm(coords, axis=1)
    sorted_indices = np.argsort(dists)
    initial_points = coords[sorted_indices[:12]]
    
    def objective(x):
        points = x.reshape(12, 2)
        angles = np.deg2rad(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = points @ normals.T
        widths = np.max(projs, axis=0) - np.min(projs, axis=0)
        S = np.max(widths) / np.sqrt(3)
        
        penalty = 0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(points[i] - points[j])
                if d < np.sqrt(3):
                    penalty += (np.sqrt(3) - d)**2
        return S + 10 * penalty

    best_S = float('inf')
    best_x = initial_points.flatten()
    
    # Multi-start
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        x0 = initial_points.flatten()
        if restart > 0:
            x0 += np.random.normal(0, 0.2, 24)
            
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3})
        
        if res.fun < best_S:
            best_S = res.fun
            best_x = res.x

    final_points = best_x.reshape(12, 2)
    
    angles = np.deg2rad(np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    projs = final_points @ normals.T
    widths = np.max(projs, axis=0) - np.min(projs, axis=0)
    final_S = np.max(widths) / np.sqrt(3)
    
    center = np.mean(final_points, axis=0)
    final_points = final_points - center
    
    inner = []
    for p in final_points:
        inner.append([p[0], p[1], 0.0])
        
    return inner, [0.0, 0.0], final_S, 0.0
