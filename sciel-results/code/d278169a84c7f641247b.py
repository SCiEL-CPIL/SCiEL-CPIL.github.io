"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    sqrt3 = np.sqrt(3)
    
    def axial_to_xy(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # Hexagonal cluster: center + 6 + 4
    positions = [(0, 0)]
    for k in range(6):
        angle = k * np.pi / 3
        positions.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
    for q, r in [(2, 0), (2, -1), (-2, 0), (-2, 1)]:
        positions.append(axial_to_xy(q, r))
    
    positions = np.array(positions)
    angles = np.zeros(N)
    
    # Get all vertices
    all_points = []
    for i in range(N):
        cx, cy = positions[i]
        rot = angles[i]
        for k in range(6):
            angle = rot + k * np.pi / 3
            all_points.append((cx + np.cos(angle), cy + np.sin(angle)))
    all_points = np.array(all_points)
    
    best_S = float('inf')
    best_c = (0, 0)
    best_alpha = 0
    
    # Coarse search
    for alpha_deg in range(0, 30, 2):
        if time.time() - start_time > 100:
            break
        alpha = np.radians(alpha_deg)
        normals = np.array([(np.cos(alpha + np.pi/6 + k*np.pi/3), 
                             np.sin(alpha + np.pi/6 + k*np.pi/3)) for k in range(6)])
        
        def objective(c):
            cvec = np.array(c)
            dots = (all_points - cvec) @ normals.T
            return np.max(dots)
        
        c0 = all_points.mean(axis=0)
        res = minimize(objective, c0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        S_val = res.fun / (sqrt3 / 2)
        
        if S_val < best_S:
            best_S = S_val
            best_c = (res.x[0], res.x[1])
            best_alpha = alpha
    
    # Fine search
    for alpha_deg in np.arange(best_alpha - 2, best_alpha + 2, 0.1):
        if time.time() - start_time > 110:
            break
        alpha = np.radians(alpha_deg)
        normals = np.array([(np.cos(alpha + np.pi/6 + k*np.pi/3), 
                             np.sin(alpha + np.pi/6 + k*np.pi/3)) for k in range(6)])
        
        def objective(c):
            cvec = np.array(c)
            dots = (all_points - cvec) @ normals.T
            return np.max(dots)
        
        c0 = all_points.mean(axis=0)
        res = minimize(objective, c0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        S_val = res.fun / (sqrt3 / 2)
        
        if S_val < best_S:
            best_S = S_val
            best_c = (res.x[0], res.x[1])
            best_alpha = alpha
    
    inner = [[positions[i][0], positions[i][1], 0.0] for i in range(N)]
    return inner, (best_c[0], best_c[1]), best_S, np.degrees(best_alpha)
# EVOLVE-BLOCK-END
