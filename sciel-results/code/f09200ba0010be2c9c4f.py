import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 20.0

def construct():
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Hexagonal close-packing arrangement for 11 unit hexagons
    # Center + 6 in first ring + 4 in second ring (selected for symmetry)
    # Using axial coordinates (q, r) for hexagonal grid
    axial = [
        (0, 0),                          # center
        (1, 0), (0, 1), (-1, 1),         # first ring (3 of 6)
        (-1, 0), (0, -1), (1, -1),       # first ring (remaining 3)
        (2, -1), (-2, 1), (2, 0), (-1, 2) # second ring: 4 selected for balance
    ]
    
    # Convert axial to pixel coordinates
    # For unit hexagons (side length 1), the distance between adjacent centers is 2*APOTHEM = sqrt(3)
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    
    pts = np.array(pts)
    
    # Center the configuration at origin
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    
    # Function to compute minimal side length S for a given outer rotation beta (radians)
    def calc_S(beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        # Outer hexagon normals (directions perpendicular to each side)
        # For a regular hexagon with apothem a, the distance from center to each side is a
        # If we rotate by beta, the normal directions are beta + k*pi/3 for k=0..5
        # Actually, for a flat-top hexagon, normals are at 0, 60, 120, 180, 240, 300 degrees
        # The apothem of the outer hexagon with side S is S*APOTHEM
        # So we need: max over all normals of (max projection of inner points onto that normal) <= S*APOTHEM
        # Therefore: S >= max_over_normals (max_projection / APOTHEM)
        
        max_S = 0.0
        for k in range(6):
            ang = beta + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projections = pts[:, 0] * nx + pts[:, 1] * ny
            abs_proj = np.max(np.abs(projections))
            required_S = abs_proj / APOTHEM
            if required_S > max_S:
                max_S = required_S
        
        return max_S

    # Minimize over rotation in [0, pi/6] (hexagon has 6-fold symmetry)
    best_beta = 0.0
    best_S = calc_S(0.0)
    
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(calc_S, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 200, 'xatol': 1e-10})
            if res.fun < best_S:
                best_beta = res.x
                best_S = res.fun
    except Exception:
        pass
    
    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation

