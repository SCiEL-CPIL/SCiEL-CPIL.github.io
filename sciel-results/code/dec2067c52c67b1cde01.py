import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0 # Safety margin before 150s

def _hex_vertices(cx, cy, angle_deg):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg.
    """
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg):
    """
    Calculate the minimum side length S of the outer hexagon centered at (0,0)
    given inner hexagons at centers with given angles.
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    # Normals to the sides of the outer hexagon
    # If outer hex is rotated by theta, its sides are at theta + 30, theta + 90, etc.
    # The normals point to the midpoints of the sides.
    normals = []
    for k in range(6):
        alpha = theta_rad + np.deg2rad(30 + k * 60)
        normals.append((np.cos(alpha), np.sin(alpha)))
    
    max_proj = 0.0
    for (cx, cy), ang in zip(inner_centers, inner_angles):
        verts = _hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _sat_overlap(p1, p2):
    """
    Check if two convex polygons (arrays of (x,y)) overlap using Separating Axis Theorem.
    """
    axes = []
    for poly in [p1, p2]:
        for i in range(len(poly)):
            ax = poly[(i+1) % len(poly)][0] - poly[i][0]
            ay = poly[(i+1) % len(poly)][1] - poly[i][1]
            nx = -ay
            ny = ax
            axes.append((nx, ny))
    
    for nx, ny in axes:
        min1 = min(v[0]*nx + v[1]*ny for v in p1)
        max1 = max(v[0]*nx + v[1]*ny for v in p1)
        min2 = min(v[0]*nx + v[1]*ny for v in p2)
        max2 = max(v[0]*nx + v[1]*ny for v in p2)
        
        if max1 + 1e-9 < min2 or max2 + 1e-9 < min1:
            return False
            
    return True

def construct():
    start_time = time.time()
    
    # Lattice basis for hexagons of side 1
    # Distance between centers of adjacent hexagons is sqrt(3)
    d = np.sqrt(3)
    
    # Base configuration: Center + 6 neighbors (Ring 1)
    # Angles 0, 60, ...
    base_centers = [
        (0.0, 0.0),
        (d, 0.0),
        (d/2, 1.5),
        (-d/2, 1.5),
        (-d, 0.0),
        (-d/2, -1.5),
        (d/2, -1.5)
    ]
    base_angles = [0.0] * 7
    
    # Ring 2 candidates (Type B: distance 3, angles 30, 90, ...)
    # We pick 4 symmetric ones: 30, 150, 210, 330 degrees
    r2_dist = 3.0
    ring2_angles = [30, 150, 210, 330]
    extra_centers = []
    extra_angles = []
    for ang_deg in ring2_angles:
        rad = np.deg2rad(ang_deg)
        cx = r2_dist * np.cos(rad)
        cy = r2_dist * np.sin(rad)
        extra_centers.append((cx, cy))
        extra_angles.append(0.0) # Initial angle
    
    all_centers = base_centers + extra_centers
    all_angles = base_angles + extra_angles
    
    # Initial outer angle
    outer_angle = 0.0
    
    # Objective function for optimization
    # We optimize the positions and angles of the 4 extra hexagons (indices 7-10)
    # and the outer hexagon angle.
    # Total parameters: 4*2 (pos) + 4*1 (ang) + 1 (outer) = 13
    
    def obj(params):
        # Unpack parameters
        # params[0:8] are x, y for 4 hexagons
        # params[8:12] are angles for 4 hexagons
        # params[12] is outer angle
        
        centers = np.array(base_centers, dtype=float)
        angles = np.array(base_angles, dtype=float)
        
        for k in range(4):
            centers[7+k] = [params[2*k], params[2*k+1]]
            angles[7+k] = params[8+k]
            
        outer_ang = params[12]
        
        # Check for overlaps
        # Only need to check pairs involving the moved hexagons?
        # No, moving one might push another? No, they are independent.
        # But they might overlap with each other or with the base.
        # We check all pairs to be safe.
        polys = [_hex_vertices(c[0], c[1], a) for c, a in zip(centers, angles)]
        
        for i in range(N):
            for j in range(i+1, N):
                if _sat_overlap(polys[i], polys[j]):
                    return 1e10 + (time.time() - start_time) # Penalty
            
        S = _min_outer_side(list(centers), list(angles), outer_ang)
        return S
    
    # Initial guess
    x0 = []
    for k in range(4):
        x0.append(extra_centers[k][0])
        x0.append(extra_centers[k][1])
    for k in range(4):
        x0.append(extra_angles[k])
    x0.append(outer_angle)
    
    best_S = float('inf')
    best_config = None
    
    # Quick check of initial config
    init_S = obj(x0)
    if init_S < 1e9:
        best_S = init_S
        best_config = x0.copy()
        
    # Optimization
    try:
        result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_S and result.fun < 1e9:
            best_S = result.fun
            best_config = result.x
            
    except Exception:
        pass
        
    if best_config is None:
        # Fallback to a valid but potentially suboptimal configuration
        # Just return the initial lattice config
        inner = [[c[0], c[1], a] for c, a in zip(all_centers, all_angles)]
        return inner, (0.0, 0.0), _min_outer_side(all_centers, all_angles, 0.0), 0.0

    # Extract final configuration
    centers = np.array(base_centers, dtype=float)
    angles = np.array(base_angles, dtype=float)
    
    for k in range(4):
        centers[7+k] = [best_config[2*k], best_config[2*k+1]]
        angles[7+k] = best_config[8+k]
        
    outer_ang = best_config[12]
    
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
