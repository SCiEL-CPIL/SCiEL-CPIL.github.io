import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 148.0

def _unit_hex_vertices():
    """Vertices of a unit regular hexagon centered at (0,0), aligned with x-axis."""
    pts = []
    for k in range(6):
        a = math.radians(60 * k)
        pts.append((math.cos(a), math.sin(a)))
    return np.array(pts)

UNIT_HEX = _unit_hex_vertices()

def _get_poly(cx, cy):
    """Get vertices of a unit hexagon at (cx, cy) with rotation 0."""
    return UNIT_HEX + np.array([cx, cy])

def _min_outer_side_fast(centers):
    """
    Calculate min side length of outer hexagon (centered at 0,0, rot 0) 
    containing hexagons at given centers.
    centers: (N, 2) array
    """
    angles = [math.radians(30 + k * 60) for k in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in angles])
    
    max_proj_units = np.array([np.max(UNIT_HEX @ normals[k]) for k in range(6)])
    
    center_projs = centers @ normals.T  # (N, 6)
    
    max_over_j = np.max(center_projs, axis=0) + max_proj_units
    
    r_min = np.max(max_over_j)
    
    if r_min < 0:
        return 0.0
        
    s = 2.0 * r_min / math.sqrt(3.0)
    return s

def _sat_penetration(A, B):
    """
    Fast SAT penetration depth for two convex polygons.
    A, B: (6, 2) arrays of vertices.
    """
    min_overlap = float('inf')
    
    normals = np.array([[math.cos(math.radians(30 + k*60)), math.sin(math.radians(30 + k*60))] for k in range(6)])
    
    projA = A @ normals.T
    projB = B @ normals.T
    
    minA = np.min(projA, axis=0)
    maxA = np.max(projA, axis=0)
    minB = np.min(projB, axis=0)
    maxB = np.max(projB, axis=0)
    
    for k in range(6):
        if maxA[k] < minB[k] or maxB[k] < minA[k]:
            return 0.0
        
        overlap = min(maxA[k], maxB[k]) - max(minA[k], minB[k])
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def _overlap_penalty(centers):
    """
    Calculate total squared penetration depth between all pairs of hexagons.
    centers: (N, 2) array
    """
    penalty = 0.0
    polys = [_get_poly(centers[i][0], centers[i][1]) for i in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            pen = _sat_penetration(polys[i], polys[j])
            if pen > 0:
                penalty += pen * pen
    return penalty

def _objective(centers):
    """
    Objective function for the fixed-rotation packing.
    centers: (N, 2) array
    """
    s = _min_outer_side_fast(centers)
    pen = _overlap_penalty(centers)
    
    return s + 100.0 * pen

def _get_seed():
    d = math.sqrt(3.0) * 1.01
    pts = [[0.0, 0.0]]
    for k in range(6):
        a = math.radians(60 * k)
        pts.append([d * math.cos(a), d * math.sin(a)])
    for k in range(4):
        a = math.radians(30 + k * 90)
        r = 2.0 * d
        pts.append([r * math.cos(a), r * math.sin(a)])
    return np.array(pts)

def construct():
    best_centers = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0
    
    initial_seed = _get_seed()
    initial_score = _objective(initial_seed)
    
    if initial_score < best_score:
        best_centers = initial_seed.copy()
        best_score = initial_score
    
    T = 1.0
    T_min = 1e-6
    cooling = 0.99
    current_centers = initial_seed.copy()
    current_score = initial_score
    
    while time.time() < budget:
        i = random.randint(0, N - 1)
        delta_x = random.gauss(0, 0.1)
        delta_y = random.gauss(0, 0.1)
        
        new_centers = current_centers.copy()
        new_centers[i][0] += delta_x
        new_centers[i][1] += delta_y
        
        new_score = _objective(new_centers)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_centers = new_centers
            current_score = new_score
            
            if current_score < best_score:
                best_centers = current_centers.copy()
                best_score = current_score
        
        T *= cooling
        if T < T_min:
            T = 0.5
            for k in range(N):
                current_centers[k][0] += random.gauss(0, 0.5)
                current_centers[k][1] += random.gauss(0, 0.5)
            current_score = _objective(current_centers)
            if current_score < best_score:
                best_centers = current_centers.copy()
                best_score = current_score

    if best_centers is not None:
        try:
            res = minimize(
                lambda x: _objective(x.reshape(N, 2)),
                best_centers.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res.fun < best_score:
                best_centers = res.x.reshape(N, 2)
                best_score = res.fun
        except Exception:
            pass

    inner = [[float(best_centers[i][0]), float(best_centers[i][1]), 0.0] for i in range(N)]
    
    s = _min_outer_side_fast(best_centers)
    
    polys = [_get_poly(inner[i][0], inner[i][1]) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if _sat_penetration(polys[i], polys[j]) > 1e-7:
                valid = False
                break
        if not valid: break
    
    if not valid:
        seed_inner = [[p[0], p[1], 0.0] for p in _get_seed()]
        seed_s = _min_outer_side_fast(np.array(seed_inner))
        return seed_inner, (0.0, 0.0), seed_s, 0.0
        
    return inner, (0.0, 0.0), s, 0.0
