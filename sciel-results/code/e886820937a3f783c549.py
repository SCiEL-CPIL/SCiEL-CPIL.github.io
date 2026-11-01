import math
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
MIN_DIST = math.sqrt(3) - 1e-7 
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def compute_min_enclosing_side(pts, center, angle):
    """
    Compute the side length S of the smallest regular hexagon
    centered at 'center' with rotation 'angle' that contains all points in 'pts'.
    Assumes inner hexagons are aligned with the global axes (angle 0).
    """
    normals = []
    for k in range(6):
        ang = angle + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    D = 0.0
    for nx, ny in normals:
        for px, py in pts:
            proj = (px - center[0]) * nx + (py - center[1]) * ny
            abs_proj = abs(proj)
            if abs_proj > D:
                D = abs_proj
                
    S = 1.0 + D / APOTHEM
    return S

def objective(params):
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 1e100
        
    pts = []
    for i in range(N):
        x = params[2*i]
        y = params[2*i+1]
        pts.append((x, y))
    
    cx = params[22]
    cy = params[23]
    angle = params[24]
    
    # Check overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < MIN_DIST:
                # Quadratic penalty for overlap
                overlap = MIN_DIST - dist
                penalty += overlap * overlap * 100.0

    if penalty > 1e-6:
        return 1e10 + penalty

    S = compute_min_enclosing_side(pts, (cx, cy), angle)
    return S

def construct():
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    initial_pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        initial_pts.append((x, y))
        
    cx_init = np.mean([p[0] for p in initial_pts])
    cy_init = np.mean([p[1] for p in initial_pts])
    
    x0 = []
    for p in initial_pts:
        x0.append(p[0])
        x0.append(p[1])
    x0.append(cx_init)
    x0.append(cy_init)
    x0.append(0.0)
    
    bounds = [(-5, 5)] * 22 + [(-5, 5), (-5, 5), (0, math.pi/6)]
    
    best_S = float('inf')
    best_params = None
    
    # Phase 1: Differential Evolution
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res_de = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=15, 
                tol=1e-6,
                polish=False
            )
            if res_de.fun < best_S:
                best_S = res_de.fun
                best_params = res_de.x
    except Exception:
        pass
    
    # Phase 2: Local Polish with Nelder-Mead
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            res_nm = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if res_nm.fun < best_S:
                best_S = res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
            
    # Phase 3: Another polish
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            res_lbfgs = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res_lbfgs.fun < best_S:
                best_S = res_lbfgs.fun
                best_params = res_lbfgs.x
        except Exception:
            pass
            
    if best_params is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i+1]
        angle_deg = math.degrees(best_params[24])
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_params[22]), float(best_params[23])]
    side = float(best_S)
    rotation = math.degrees(best_params[24])
    
    return inner, center, side, rotation
