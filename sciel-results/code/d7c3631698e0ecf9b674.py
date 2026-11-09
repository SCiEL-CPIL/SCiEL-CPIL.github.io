"""Hexagon packing: grid-based NM with exact SAT, symmetry-reduced search."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# --- Exact geometry helpers (vectorized where possible) ---

def _hex_verts(cx, cy, angle_deg):
    """6 vertices of unit hexagon centered at (cx,cy) rotated by angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Base vertices at angles 0,60,...,300 (flat-top: vertices at those angles)
    base = np.array([[math.cos(math.radians(k*60)), math.sin(math.radians(k*60))] for k in range(6)])
    rot = base @ np.array([[ca, -sa],[sa, ca]]).T
    return rot + np.array([cx, cy])

def _hex_normals(angle_deg):
    """6 outward normal directions (unit vectors) of the hexagon edges."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    base = np.array([[math.cos(math.radians(k*60+30)), math.sin(math.radians(k*60+30))] for k in range(6)])
    rot = base @ np.array([[ca, -sa],[sa, ca]]).T
    return rot

def _sat_separation(v1, v2, n1, n2):
    """
    Compute the minimum separation (positive = disjoint, negative = overlapping)
    between two convex hexagons using SAT.
    v1,v2: (6,2) vertex arrays; n1,n2: (6,2) normal arrays.
    Returns separation distance (>=0 if disjoint).
    """
    all_normals = np.vstack([n1, n2])  # 12 normals
    min_sep = np.inf
    for nx, ny in all_normals:
        p1 = v1 @ np.array([nx, ny])
        p2 = v2 @ np.array([nx, ny])
        sep = max(p1.min() - p2.max(), p2.min() - p1.max())
        if sep < 0:
            return 0.0  # Overlapping
        if sep < min_sep:
            min_sep = sep
    return min_sep

def _outer_side_length(centers, angles, outer_angle_deg=0.0):
    """
    Compute the minimum outer hexagon side length (with given outer rotation)
    needed to contain all inner hexagons. Centers are already centered at origin.
    """
    # Outer hexagon normals (6 directions)
    oa = math.radians(outer_angle_deg)
    outer_normals = np.array([[math.cos(math.radians(k*60+30)+oa),
                                math.sin(math.radians(k*60+30)+oa)] for k in range(6)])
    
    max_proj = 0.0
    for i in range(N):
        cx, cy = centers[i]
        a = angles[i]
        verts = _hex_verts(cx, cy, a)
        for nx, ny in outer_normals:
            projs = np.abs(verts @ np.array([nx, ny]))
            if projs.max() > max_proj:
                max_proj = projs.max()
    
    # For a regular hexagon with inradius r, side = 2r/sqrt(3)
    return 2.0 * max_proj / SQRT3

def _evaluate_exact(params):
    """
    Exact evaluation: returns (outer_side, overlap_penalty).
    params: (3N,) array [x0,y0,a0, x1,y1,a1, ...]
    Centers are re-centered to mean(0,0) before evaluation.
    """
    centers = np.array([(params[i*3], params[i*3+1]) for i in range(N)])
    angles = np.array([params[i*3+2] for i in range(N)])
    
    # Re-center
    centers -= centers.mean(axis=0)
    
    # Compute vertices and normals for all hexagons
    verts_list = []
    norms_list = []
    for i in range(N):
        verts_list.append(_hex_verts(centers[i,0], centers[i,1], angles[i]))
        norms_list.append(_hex_normals(angles[i]))
    
    # Overlap check
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            sep = _sat_separation(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if sep == 0.0:
                # Overlapping: compute penetration depth as penalty
                # Use center distance as proxy
                d = np.linalg.norm(centers[i] - centers[j])
                penetration = max(0.0, SQRT3 - d)
                overlap_penalty += penetration * penetration + 0.1
            elif sep < 0.01:
                # Very close: small penalty to encourage gap
                overlap_penalty += (0.01 - sep) * 10.0
    
    # Outer side length (fix outer angle to 0 for symmetry reduction)
    side = _outer_side_length(centers, angles, 0.0)
    
    return side + overlap_penalty * 5.0

def _is_valid(params):
    """Check if configuration is valid (no overlaps)."""
    centers = np.array([(params[i*3], params[i*3+1]) for i in range(N)])
    angles = np.array([params[i*3+2] for i in range(N)])
    centers -= centers.mean(axis=0)
    
    verts_list = []
    norms_list = []
    for i in range(N):
        verts_list.append(_hex_verts(centers[i,0], centers[i,1], angles[i]))
        norms_list.append(_hex_normals(angles[i]))
    
    for i in range(N):
        for j in range(i+1, N):
            sep = _sat_separation(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if sep < 1e-9:
                return False
    return True

def _get_side_length(params):
    """Get the outer side length for a valid configuration."""
    centers = np.array([(params[i*3], params[i*3+1]) for i in range(N)])
    angles = np.array([params[i*3+2] for i in range(N)])
    centers -= centers.mean(axis=0)
    return _outer_side_length(centers, angles, 0.0)

# --- Initial configurations ---

def _honeycomb_init():
    """Standard honeycomb arrangement of 11 hexagons."""
    coords = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    params = np.zeros(3*N)
    for i, (q, r) in enumerate(coords):
        x = SQRT3 * (q + r/2.0)
        y = 1.5 * r
        params[i*3] = x
        params[i*3+1] = y
        params[i*3+2] = 0.0
    return params

def _ring_init():
    """Central hexagon + 6 ring + 4 more."""
    params = np.zeros(3*N)
    # Center
    params[0] = 0.0; params[1] = 0.0; params[2] = 0.0
    # Ring of 6
    for k in range(6):
        idx = k + 1
        angle = math.radians(k*60)
        r = SQRT3
        params[idx*3] = r * math.cos(angle)
        params[idx*3+1] = r * math.sin(angle)
        params[idx*3+2] = 0.0
    # 4 more: placed at 120° intervals, further out
    for k in range(4):
        idx = k + 7
        angle = math.radians(k*120 + 30)
        r = 2.5 * SQRT3
        params[idx*3] = r * math.cos(angle)
        params[idx*3+1] = r * math.sin(angle)
        params[idx*3+2] = 0.0
    return params

def _compact_init():
    """Compact cluster: 3 rows of 4,3,4 with offsets."""
    params = np.zeros(3*N)
    positions = []
    # Row 1 (top): 4 hexagons
    for k in range(4):
        positions.append((k*SQRT3 - 1.5*SQRT3, 1.5))
    # Row 2 (middle): 3 hexagons
    for k in range(3):
        positions.append((k*SQRT3 - SQRT3, 0.0))
    # Row 3 (bottom): 4 hexagons
    for k in range(4):
        positions.append((k*SQRT3 - 1.5*SQRT3, -1.5))
    
    for i, (x, y) in enumerate(positions[:N]):
        params[i*3] = x
        params[i*3+1] = y
        params[i*3+2] = 0.0
    return params

def _random_init(seed=None):
    """Random initial configuration within a reasonable bounding box."""
    rng = random.Random(seed)
    params = np.zeros(3*N)
    for i in range(N):
        params[i*3] = rng.uniform(-3.0, 3.0)
        params[i*3+1] = rng.uniform(-3.0, 3.0)
        params[i*3+2] = rng.uniform(0.0, 60.0)
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_score = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    inits = [
        _honeycomb_init(),
        _ring_init(),
        _compact_init(),
        _random_init(42),
        _random_init(123),
        _random_init(777),
        _random_init(2024),
        _random_init(999),
    ]
    
    # Grid of outer angles to try (but we fix to 0 during optimization for speed)
    # We'll optimize outer angle separately at the end
    
    for init_idx, init_params in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        current_params = init_params.copy()
        current_score = _evaluate_exact(current_params)
        
        # Nelder-Mead optimization
        if time.time() > deadline - 10.0:
            break
        
        def objective(p):
            return _evaluate_exact(p)
        
        try:
            res = minimize(objective, current_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 
                                   'adaptive': True, 'initial_simplex': 0.1})
            if res.fun < current_score:
                current_score = res.fun
                current_params = res.x
        except Exception:
            pass
        
        # Second pass with tighter tolerance
        if time.time() > deadline - 8.0:
            break
        
        try:
            res = minimize(objective, current_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < current_score:
                current_score = res.fun
                current_params = res.x
        except Exception:
            pass
        
        if current_score < best_score:
            best_score = current_score
            best_params = current_params.copy()
    
    # If no valid result yet, use honeycomb as fallback
    if best_params is None:
        best_params = _honeycomb_init()
    
    # Final: optimize outer hexagon angle for the best inner configuration
    centers = np.array([(best_params[i*3], best_params[i*3+1]) for i in range(N)])
    angles = np.array([best_params[i*3+2] for i in range(N)])
    centers -= centers.mean(axis=0)
    
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in np.arange(0, 60, 0.5):
        side = _outer_side_length(centers, angles, oa_deg)
        if side < best_side:
            best_side = side
            best_oa = oa_deg
    
    # Fine-tune outer angle
    for oa_deg in np.arange(best_oa - 0.5, best_oa + 0.5, 0.05):
        side = _outer_side_length(centers, angles, oa_deg)
        if side < best_side:
            best_side = side
            best_oa = oa_deg
    
    # Build inner list
    inner_list = []
    for i in range(N):
        inner_list.append([centers[i,0], centers[i,1], angles[i]])
    
    return (inner_list, (0.0, 0.0), best_side, best_oa)
