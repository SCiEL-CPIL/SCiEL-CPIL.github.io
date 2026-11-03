import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    # Lattice vectors for honeycomb packing of unit hexagons
    # Distance between centers is sqrt(3)
    # We scale by 1.01 to ensure strict disjointness
    scale = 1.01
    u = np.array([np.sqrt(3), 0.0]) * scale
    v = np.array([np.sqrt(3)/2.0, 1.5]) * scale
    
    points = []
    for i in range(4):
        for j in range(3):
            p = i * u + j * v
            points.append(p)
            
    points = np.array(points)
    
    # Center the points around the origin
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Function to compute the required side length S for a given rotation phi and center C
    def objective(params):
        phi, cx, cy = params
        C = np.array([cx, cy])
        # Normals of the outer hexagon
        max_proj = 0.0
        for k in range(6):
            angle = phi + k * np.pi / 3.0
            n = np.array([np.cos(angle), np.sin(angle)])
            # Project all points onto n
            projs = np.dot(points - C, n)
            # The max absolute projection is the required apothem in this direction
            max_abs_proj = np.max(np.abs(projs))
            if max_abs_proj > max_proj:
                max_proj = max_abs_proj
        
        # Apothem a = max_proj
        # Side length S = 2 * a / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)
        
    # Initial guess
    # Try a few rotations to find a good starting point
    best_result = None
    best_phi = 0.0
    for phi_test in np.linspace(0, np.pi/6, 10):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        res = minimize(objective, [phi_test, 0.0, 0.0], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
        if best_result is None or res.fun < best_result.fun:
            best_result = res
            best_phi = res.x[0]
            
    if best_result is None:
        # Fallback
        return [[p[0], p[1], 30.0] for p in points], [0, 0], 10.0, 0.0

    # Refine with the best phi
    if time.time() - START_TIME < TIME_LIMIT - 5:
        res = minimize(objective, best_result.x, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    else:
        res = best_result

    phi_opt, cx_opt, cy_opt = res.x
    S_opt = res.fun
    
    # The inner hexagons are at 'points' with rotation 30.0
    # 30.0 degrees aligns flat sides with the lattice vectors (assuming 0 is vertex-on-x)
    inner = [[p[0], p[1], 30.0] for p in points]
    center = [cx_opt, cy_opt]
    side = S_opt
    angle = phi_opt
    
    return inner, center, side, angle
