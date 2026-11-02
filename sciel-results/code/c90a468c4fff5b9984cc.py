import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Candidates: Compact configurations of 11 hexagons
    candidates = []
    
    # 1+6+4 (Flower with 4 extra)
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    
    # 3x3 + 2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    
    # 3x3 + 2 (different placement)
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    
    # 4x3 - 1
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    
    # 4x3 - 2 (corners)
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    
    # Star
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    
    # Asymmetric Star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    
    # Compact Zigzag
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    
    # 3x4 - 1
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    
    # 3x4 - 2
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])

    # Additional diverse shapes
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    def calc_D_for_center(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        return np.max(np.abs(projs))

    def objective_center(params, pts, beta):
        cx, cy = params
        return calc_D_for_center(pts, cx, cy, beta)

    def optimize_shape(pts):
        best_D = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Coarse grid search for beta
        for i in range(13): # 0 to pi/6 step pi/72
            beta = i * (math.pi / 6) / 12
            # Minimize over center
            # Initial guess: centroid
            cx0, cy0 = np.mean(pts, axis=0)
            res = minimize(objective_center, [cx0, cy0], args=(pts, beta), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_D:
                best_D = res.fun
                best_beta = beta
                best_center = (res.x[0], res.x[1])
        
        # Refine beta
        def obj_beta(beta):
            # Re-optimize center for this beta
            cx0, cy0 = best_center
            res = minimize(objective_center, [cx0, cy0], args=(pts, beta), method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
            return res.fun
        
        res_beta = minimize_scalar(obj_beta, bounds=(max(0, best_beta - 0.1), min(math.pi/6, best_beta + 0.1)), method='bounded')
        if res_beta.fun < best_D:
            best_D = res_beta.fun
            best_beta = res_beta.x
            # Update center
            cx0, cy0 = best_center
            res_c = minimize(objective_center, [cx0, cy0], args=(pts, best_beta), method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            best_center = (res_c.x[0], res_c.x[1])
            best_D = res_c.fun

        S = (best_D + APOTHEM) / APOTHEM
        return S, best_beta, best_center

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        cx, cy = np.mean(pts, axis=0)
        beta = 0.0
        D = calc_D_for_center(pts, cx, cy, beta)
        best_global_S = (D + APOTHEM) / APOTHEM
        best_global_pts = pts
        best_global_beta = beta
        best_global_center = (cx, cy)

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
