import math
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def generate_hexagon_cells(radius):
    cells = []
    for q in range(-radius, radius + 1):
        for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
            cells.append((q, r))
    return cells

CELLS_19 = generate_hexagon_cells(2)

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def calc_enclosing_hexagon_side(pts, center, beta):
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj = 0.0
    cx, cy = center
    for nx, ny in normals:
        for px, py in pts:
            proj = abs((px - cx) * nx + (py - cy) * ny)
            if proj > max_proj:
                max_proj = proj
    
    D = max_proj
    S = (D + APOTHEM) / APOTHEM
    return S, D

def optimize_center_for_beta(pts, beta):
    centroid = (np.mean(pts[:, 0]), np.mean(pts[:, 1]))
    
    def obj_center(cx_cy):
        cx_c, cy_c = cx_cy
        S, _ = calc_enclosing_hexagon_side(pts, (cx_c, cy_c), beta)
        return S
    
    try:
        res = minimize(obj_center, centroid, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        return res.x, res.fun
    except Exception:
        return centroid, obj_center(centroid)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback
        CELLS_ORIG = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        pts = axial_to_cartesian(CELLS_ORIG)
        centroid = (np.mean(pts[:, 0]), np.mean(pts[:, 1]))
        best_S, best_center, best_beta = float('inf'), centroid, 0.0
        for i in range(6):
            beta = i * math.pi / 6
            center, S = optimize_center_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_center = center
                best_beta = beta
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(best_center[0]), float(best_center[1])], float(best_S), float(math.degrees(best_beta))

    all_pts = axial_to_cartesian(CELLS_19)
    
    best_S_global = float('inf')
    best_subset = None
    best_beta_global = 0.0
    best_center_global = (0.0, 0.0)
    
    coarse_betas = [0, 15, 30, 45]
    
    for combo in combinations(range(19), 11):
        pts = all_pts[list(combo)]
        centroid = (np.mean(pts[:, 0]), np.mean(pts[:, 1]))
        
        min_S_coarse = float('inf')
        for beta_deg in coarse_betas:
            beta = math.radians(beta_deg)
            S, _ = calc_enclosing_hexagon_side(pts, centroid, beta)
            if S < min_S_coarse:
                min_S_coarse = S
        
        if min_S_coarse < best_S_global:
            best_S_global = min_S_coarse
            best_subset = combo

    if best_subset is None:
        return construct()

    pts_best = all_pts[list(best_subset)]
    centroid_best = (np.mean(pts_best[:, 0]), np.mean(pts_best[:, 1]))
    
    best_S_refined = float('inf')
    best_beta_refined = 0.0
    best_center_refined = centroid_best
    
    for i in range(61):
        beta_deg = i
        beta = math.radians(beta_deg)
        center, S = optimize_center_for_beta(pts_best, beta)
        if S < best_S_refined:
            best_S_refined = S
            best_beta_refined = beta
            best_center_refined = center
    
    if best_S_refined < best_S_global:
        best_S_global = best_S_refined
        best_beta_global = best_beta_refined
        best_center_global = best_center_refined
    else:
        # Re-optimize center for the best coarse beta if refinement didn't help
        # This case is unlikely but safe
        min_S_coarse = float('inf')
        best_coarse_beta = 0.0
        for beta_deg in coarse_betas:
            beta = math.radians(beta_deg)
            S, _ = calc_enclosing_hexagon_side(pts_best, centroid_best, beta)
            if S < min_S_coarse:
                min_S_coarse = S
                best_coarse_beta = beta
        
        center, S = optimize_center_for_beta(pts_best, best_coarse_beta)
        if S < best_S_global:
            best_S_global = S
            best_beta_global = best_coarse_beta
            best_center_global = center

    inner = []
    for idx in best_subset:
        x, y = all_pts[idx]
        inner.append([float(x), float(y), 0.0])
    
    center = [float(best_center_global[0]), float(best_center_global[1])]
    side = float(best_S_global)
    rotation = float(math.degrees(best_beta_global))
    
    return inner, center, side, rotation
