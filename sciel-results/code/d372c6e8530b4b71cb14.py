import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0

def min_side_for_points(points, center, angle_deg):
    """Minimum side length of hexagon with given center and angle to contain all points."""
    cx, cy = center
    theta = np.radians(angle_deg)
    angles = theta + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    rel = points - np.array([cx, cy])
    dots = rel @ normals.T
    max_proj = np.max(dots, axis=1)
    return np.max(max_proj) * 2.0 / np.sqrt(3)

def objective(x, start_time):
    """
    x: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, cx, cy, side, angle]
    Total: 11*3 + 4 = 37 variables
    """
    if time.time() - start_time > DEADLINE:
        return 1e6
    
    n = 11
    centers = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
    angles = np.array([x[3*i+2] for i in range(n)])
    
    cx = x[33]
    cy = x[34]
    side = x[35]
    angle = x[36]
    
    if side < 0.1:
        return 1e6
    
    # Objective: minimize side length
    obj = side
    
    # Penalty for hexagons not being inside container
    theta = np.radians(angle)
    norm_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
    rel = centers - np.array([cx, cy])
    dots = rel @ normals.T
    max_proj = np.max(dots, axis=1)
    inradius = side * np.sqrt(3) / 2.0
    container_viol = np.maximum(0, max_proj - inradius)
    obj += 100.0 * np.max(container_viol)
    
    # Penalty for overlapping hexagons (conservative: d >= 2)
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.0:
                overlap_pen += (2.0 - d) ** 2
    obj += 50.0 * overlap_pen
    
    return obj

def construct():
    start_time = time.time()
    
    n = 11
    
    # Initial configuration: hexagonal lattice
    positions = []
    angles_init = []
    
    # Center
    positions.append([0.0, 0.0])
    angles_init.append(0.0)
    
    # 6 around center at distance 2
    for k in range(6):
        ang = k * np.pi / 3
        positions.append([2.0 * np.cos(ang), 2.0 * np.sin(ang)])
        angles_init.append(0.0)
    
    # 4 more at distance ~3.5
    for k in range(4):
        ang = np.pi/6 + k * np.pi/3
        positions.append([3.5 * np.cos(ang), 3.5 * np.sin(ang)])
        angles_init.append(0.0)
    
    positions = np.array(positions)
    angles_init = np.array(angles_init)
    
    # Estimate container
    cx, cy = 0.0, 0.0
    side_est = min_side_for_points(positions, (cx, cy), 0.0) * 1.1
    angle_est = 0.0
    
    x0 = []
    for i in range(n):
        x0.extend([positions[i][0], positions[i][1], angles_init[i]])
    x0.extend([cx, cy, side_est, angle_est])
    x0 = np.array(x0)
    
    best_x = x0.copy()
    best_obj = objective(x0, start_time)
    
    # Try multiple starts
    np.random.seed(42)
    n_starts = 8
    
    for start in range(n_starts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        if start == 0:
            x_curr = x0.copy()
        else:
            x_curr = x0 + np.random.randn(len(x0)) * 0.3
            x_curr[35] = max(2.0, x_curr[35])
        
        try:
            result = minimize(
                lambda x: objective(x, start_time), x_curr, method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Extract best solution
    centers = []
    angles_deg = []
    for i in range(n):
        centers.append([best_x[3*i], best_x[3*i+1]])
        angles_deg.append(np.degrees(best_x[3*i+2]))
    
    cx = best_x[33]
    cy = best_x[34]
    side = best_x[35]
    angle_deg = np.degrees(best_x[36])
    
    # Recompute exact side needed
    side_exact = min_side_for_points(np.array(centers), (cx, cy), angle_deg)
    side = max(side, side_exact * 1.0001)
    
    inner = []
    for i in range(n):
        inner.append([centers[i][0], centers[i][1], angles_deg[i]])
    
    return inner, (cx, cy), side, angle_deg
