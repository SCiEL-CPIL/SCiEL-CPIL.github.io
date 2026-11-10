import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Generate FCC lattice points
    # We generate a large block and pick the 100 closest to the center
    # FCC condition: i + j + k is even
    # Range: let's use -5 to 5
    range_val = 6
    points = []
    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            for k in range(-range_val, range_val + 1):
                if (i + j + k) % 2 == 0:
                    points.append([i, j, k])
    
    points = np.array(points, dtype=float)
    
    # Center the points
    center = np.mean(points, axis=0)
    points -= center
    
    # Sort by distance to origin and take the first N
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Recenter to ensure symmetry (optional, but good)
    center = np.mean(points, axis=0)
    points -= center
    
    # 2. Calculate D_min (minimum pairwise distance)
    # For N=100, O(N^2) is fine
    D_min = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < D_min:
                D_min = d
                
    # 3. Define objective function: W_max(R)
    # We parameterize rotation by 3 Euler angles (Z-Y-Z or X-Y-Z)
    # Let's use X-Y-Z (roll, pitch, yaw)
    
    def rot_matrix(angles):
        roll, pitch, yaw = angles
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    def objective(angles):
        R = rot_matrix(angles)
        rotated = points @ R.T
        # W_max is the max absolute value of any coordinate
        W_max = np.max(np.abs(rotated))
        return W_max

    # 4. Optimize
    # Try a few initial guesses to avoid local minima
    best_w_max = float('inf')
    best_angles = None
    
    initial_guesses = [
        [0, 0, 0],
        [np.pi/4, 0, 0],
        [0, np.pi/4, 0],
        [0, 0, np.pi/4],
        [np.pi/6, np.pi/6, 0],
        [np.pi/6, 0, np.pi/6],
        [0, np.pi/6, np.pi/6]
    ]
    
    for guess in initial_guesses:
        res = minimize(objective, guess, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_w_max:
            best_w_max = res.fun
            best_angles = res.x

    R_opt = rot_matrix(best_angles)
    
    # 5. Calculate scale and final positions
    # s = 0.5 / (W_max + D_min)
    s = 0.5 / (best_w_max + D_min)
    
    centers = s * (points @ R_opt.T)
    
    return centers
