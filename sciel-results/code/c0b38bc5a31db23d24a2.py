"""
Pack 12 unit regular hexagons into a minimal regular hexagon.
Uses a smooth SDF-based objective with L-BFGS-B, then polishes with Nelder-Mead.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 135.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    theta = np.deg2rad(angle_deg)
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees + rotation
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + theta
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _max_support(pts, dirx, diry):
    """Maximum projection of points onto direction (dirx, diry)."""
    return np.max(pts[:, 0] * dirx + pts[:, 1] * diry)

def _bounding_side_hex(inner_verts, outer_center, outer_side, outer_angle_deg):
    """
    Compute the minimum side length of an outer regular hexagon (center, rotation given)
    that contains all inner hexagon vertices.
    A regular hexagon with flat-top orientation has 6 sides. The support function
    in direction perpendicular to each side must be <= side_length.
    
    For a regular hexagon centered at (ox, oy) with side s and rotation phi,
    the 6 outward normals are at angles phi + 0, phi+60, ..., phi+300.
    The constraint is: max_i (dot(v_i - c, n_j)) <= s * cos(0) = s for each normal n_j.
    Actually for a regular hexagon, the distance from center to each side is s (apothem = s).
    Wait: apothem = s * sqrt(3)/2 for a regular hexagon with side s.
    
    Let me reconsider. A regular hexagon with side length s has:
    - circumradius (center to vertex) = s
    - inradius (center to side midpoint) = s * sqrt(3)/2
    
    The 6 sides are perpendicular to directions at angles 0, 60, ..., 300 (for flat-top).
    For each side direction n_j, the constraint is:
      max_i dot(v_i - center, n_j) <= s * sqrt(3)/2
    
    So the minimum s needed is: max_j [ max_i dot(v_i - center, n_j) ] / (sqrt(3)/2)
    """
    theta = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)] for k in range(6)])
    ox, oy = outer_center
    centered = inner_verts - np.array([ox, oy])
    # For each normal, compute max projection
    projections = centered @ normals.T  # shape (n_points, 6)
    max_projs = np.max(projections, axis=0)  # shape (6,)
    # Required side length: max_proj / (sqrt(3)/2)
    min_side = np.max(max_projs) / (np.sqrt(3) / 2.0)
    return max(min_side, 0.0)

def _hexagon_distance_sdf(p, hex_center, hex_side, hex_angle_deg):
    """
    Signed distance from point p to a regular hexagon (negative inside).
    We use a smooth approximation via the support function approach.
    For optimization purposes, we'll use a penalty-based approach instead.
    """
    pass

def _pairwise_overlap_penalty(centers, angles, sides):
    """
    Smooth penalty for pairwise overlaps between hexagons.
    We approximate the distance between two hexagons using their centers and a
    smooth function. For hexagons, we can use the minimum distance between
    their boundaries. A simpler approach: use the distance between centers minus
    the sum of their 'radii' (circumradii = side length for unit hex).
    
    For a more accurate approach, we compute the actual minimum distance between
    the two hexagons by checking all edge-vertex and vertex-vertex pairs, but that's
    non-smooth. Instead, we use a smooth surrogate: the distance between centers
    minus (r1 + r2) where r is the circumradius, with a soft threshold.
    """
    n = len(centers)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            # Circumradius of each hexagon = side length
            min_dist = sides[i] + sides[j]
            # Smooth overlap: if dist < min_dist, penalty grows
            # Use softplus-like function
            overlap = min_dist - dist
            if overlap > 0:
                penalty += overlap * overlap
    return penalty

def _objective(params):
    """
    Objective: minimize outer hexagon side length + overlap penalties.
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, ox, oy, outer_angle]
    Total: 12*3 + 3 = 39 parameters
    """
    t0 = time.time()
    
    centers = np.array([params[3*i:3*i+2] for i in range(N)])
    angles = np.array([params[3*i+2] for i in range(N)])
    ox, oy = params[3*N], params[3*N+1]
    outer_angle = params[3*N+2]
    
    sides = np.ones(N)  # unit hexagons
    
    # Compute all vertices of inner hexagons
    all_verts = []
    for i in range(N):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], sides[i], angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # shape (72, 2)
    
    # Compute required outer side length
    outer_side = _bounding_side_hex(all_verts, (ox, oy), 0.0, outer_angle)
    
    # Pairwise overlap penalty
    overlap_pen = _pairwise_overlap_penalty(centers, angles, sides)
    
    # Combined objective
    obj = outer_side + 10.0 * overlap_pen
    
    return obj

def _objective_grad(params):
    """Numerical gradient (fallback if analytical is too complex)."""
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = _objective(params)
    for i in range(len(params)):
        params[i] += eps
        f1 = _objective(params)
        params[i] -= eps
        grad[i] = (f1 - f0) / eps
    return grad

def construct():
    start_time = time.time()
    
    # Initial guess: arrange hexagons in a honeycomb-like pattern
    # 12 hexagons: 3 rows of 4, or 4 rows of 3, or a 3-4-3 pattern
    # Let's try a 3-4-3 arrangement (like a larger hexagon shape)
    
    # Honeycomb packing: hexagons with flat-top orientation
    # Horizontal spacing: 2 * side * cos(30) = sqrt(3)
    # Vertical spacing: 1.5 * side
    # Actually for flat-top hexagons:
    # width = 2*side, height = sqrt(3)*side
    # Horizontal neighbor distance = 1.5 * side (for pointy-top) or 2*side*cos(30)=sqrt(3) for flat-top
    
    # Let's use pointy-top hexagons for honeycomb:
    # Vertical spacing = 1.5 * side
    # Horizontal spacing = sqrt(3) * side
    # Offset every other row by sqrt(3)/2
    
    side = 1.0
    sqrt3 = np.sqrt(3)
    
    # 3-4-3 pattern (rows of 3, 4, 3 hexagons)
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    
    initial_centers = []
    initial_angles = []
    
    # Pointy-top hexagons: angle = 90 degrees (vertices at top/bottom)
    # Actually let's use 0 degrees for flat-top and see
    
    rows = [3, 4, 3]
    y_offset = 0.0
    for r, count in enumerate(rows):
        for c in range(count):
            x = (c - (count - 1) / 2.0) * sqrt3
            y = y_offset + (r * 1.5)
            initial_centers.append([x, y])
            initial_angles.append(0.0)
        y_offset += 1.5
    
    # This gives 10 hexagons, need 12. Let's use 4-4-4 or 3-4-3-2 or similar.
    # Try 4-4-4:
    initial_centers = []
    initial_angles = []
    rows = [4, 4, 4]
    y_offset = 0.0
    for r, count in enumerate(rows):
        for c in range(count):
            x = (c - (count - 1) / 2.0) * sqrt3
            y = y_offset + (r * 1.5)
            initial_centers.append([x, y])
            initial_angles.append(0.0)
        y_offset += 1.5
    
    initial_centers = np.array(initial_centers)
    initial_angles = np.array(initial_angles)
    
    # Build initial params
    params = np.zeros(3 * N + 3)
    for i in range(N):
        params[3*i] = initial_centers[i, 0]
        params[3*i+1] = initial_centers[i, 1]
        params[3*i+2] = initial_angles[i]
    params[3*N] = 0.0   # outer center x
    params[3*N+1] = 0.0  # outer center y
    params[3*N+2] = 0.0  # outer angle
    
    best_params = params.copy()
    best_obj = _objective(params)
    
    # L-BFGS-B optimization
    try:
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    # Nelder-Mead polish
    if time.time() - start_time < DEADLINE - 10:
        try:
            result2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Extract final solution
    centers = np.array([best_params[3*i:3*i+2] for i in range(N)])
    angles = np.array([best_params[3*i+2] for i in range(N)])
    ox, oy = best_params[3*N], best_params[3*N+1]
    outer_angle = best_params[3*N+2]
    
    # Compute final outer side
    all_verts = []
    for i in range(N):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    outer_side = _bounding_side_hex(all_verts, (ox, oy), 0.0, outer_angle)
    
    # Build inner list
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(angles[i])] for i in range(N)]
    
    return inner, (float(ox), float(oy)), float(outer_side), float(outer_angle)
