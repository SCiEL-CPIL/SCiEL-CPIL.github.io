import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotated."""
    a = np.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 from center, distance = side
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + a
    r = side
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    return np.column_stack([xs, ys])

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, angle."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Regular hexagon with side s has vertices at distance s from center
    # The hexagon is the intersection of 6 half-planes
    # For a hexagon with "pointy" top (vertex at angle 90), the constraints are:
    # Actually, our hex has vertices at angles 0,60,... so it's "flat" on sides
    # Let's use the standard: hexagon with vertices at angle k*60 deg, radius = side
    # The hexagon is: |x| <= side, |y| <= side*sin(60), |x*cos(60)+y*sin(60)| <= side*sin(60)... 
    # Simpler: use the 6 normal vectors
    normals = []
    for k in range(6):
        ang = np.radians(60 * k + 30)  # normals point to edge midpoints
        normals.append((np.cos(ang), np.sin(ang)))
    for nx, ny in normals:
        dist = lx * nx + ly * ny
        if dist > side * np.cos(np.pi / 6):  # apothem = side * cos(30 deg)
            return False
    return True

def hexagon_overlap(v1, v2, tol=1e-10):
    """Check if two hexagons (each 6 vertices) overlap using SAT (Separating Axis Theorem)."""
    # Collect all edge normals from both hexagons
    axes = []
    for v in (v1, v2):
        for i in range(6):
            j = (i + 1) % 6
            edge = v[j] - v[i]
            # Normal to edge
            nx, ny = -edge[1], edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-15:
                axes.append((nx/norm, ny/norm))
    
    for ax, ay in axes:
        # Project both hexagons onto this axis
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # Check for gap
        if max1 + tol < min2 or max2 + tol < min1:
            return False  # Separated, no overlap
    return True  # No separating axis found, they overlap

def all_non_overlapping(inner_hexes, tol=1e-8):
    """Check that all pairs of inner hexagons don't overlap."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            if hexagon_overlap(inner_hexes[i], inner_hexes[j], tol):
                return False
    return True

def all_inside_container(inner_hexes, outer_cx, outer_cy, outer_side, outer_angle, tol=1e-8):
    """Check that all inner hexagons are fully inside the outer hexagon."""
    for hex_v in inner_hexes:
        for px, py in hex_v:
            if not point_in_hexagon(px, py, outer_cx, outer_cy, outer_side, outer_angle):
                return False
    return True

def objective(params, outer_angle=0.0):
    """
    params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11] for 11 inner hexagons
    Returns the outer side length needed, or a large penalty if invalid.
    """
    n = 11
    inner_hexes = []
    for i in range(n):
        x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
        v = hex_vertices(x, y, 1.0, ang)
        inner_hexes.append(v)
    
    # Check non-overlap
    if not all_non_overlapping(inner_hexes):
        return 1e10
    
    # Find minimum enclosing hexagon
    # The outer hexagon must contain all inner hexagon vertices
    # For a given outer angle, the minimum side length is determined by the
    # maximum "extent" in the 6 directions of the hexagon's normals
    
    # For a regular hexagon with center (0,0) and angle theta, the constraint for
    # a point p to be inside with side s is:
    # For each of 6 edge normals n_k (at angles theta + 30 + 60k):
    #   p . n_k <= s * cos(30)  (apothem)
    # Also need to handle the full hexagon shape
    
    # Actually, for a regular hexagon, the width in direction n_k is 2*s*cos(30) for the
    # normal directions, but we need to find the minimum s such that all points are inside.
    
    # The minimum enclosing regular hexagon (with fixed orientation) of a set of points:
    # For each of the 6 edge normal directions, the farthest point in that direction
    # determines the required apothem.
    
    all_points = np.vstack(inner_hexes)  # (66, 2)
    
    outer_ang_rad = np.radians(outer_angle)
    apothem_required = 0.0
    for k in range(6):
        ang = outer_ang_rad + np.radians(30 + 60 * k)
        nx, ny = np.cos(ang), np.sin(ang)
        # Distance of each point from center (0,0) in direction (nx, ny)
        dists = all_points[:, 0] * nx + all_points[:, 1] * ny
        # The hexagon edge is at distance apothem = s*cos(30) from center
        # All points must satisfy: p . n_k <= apothem AND p . (-n_k) <= apothem
        # i.e., |p . n_k| <= apothem
        max_dist = np.max(np.abs(dists))
        apothem_required = max(apothem_required, max_dist)
    
    # apothem = s * cos(30 deg) => s = apothem / cos(30)
    s = apothem_required / np.cos(np.pi / 6)
    
    return s

def optimize_layout(initial_params, outer_angle=0.0, max_iter=2000):
    """Optimize the inner hexagon positions to minimize outer side length."""
    result = minimize(
        objective,
        initial_params,
        args=(outer_angle,),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    return result

def construct():
    start_time = time.time()
    
    # Initial layout: hexagonal arrangement
    # 11 hexagons: 1 center, 6 in first ring, 4 in second ring (partial)
    # For unit hexagons, center-to-center distance for non-overlapping is 2*apothem = sqrt(3)
    # Actually, for two unit hexagons side by side, the distance between centers 
    # along the "flat" direction is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    # Along the "pointy" direction it's 2
    
    # Let's use a hexagonal lattice with spacing sqrt(3) (touching)
    sp = np.sqrt(3)  # center-to-center distance for touching hexagons
    
    # Positions in hex grid (using axial coordinates)
    # Center: (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: we need 4 more
    
    # Hex grid positions (in units of sp):
    # Using offset coordinates
    positions = [
        (0, 0),          # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (-1, 2), (-2, 1)  # 4 from ring 2
    ]
    
    # Convert axial to cartesian (for flat-top hexagons)
    # x_cart = sp * (q + r/2)
    # y_cart = sp * (sqrt(3)/2) * r
    initial_params = []
    for q, r in positions:
        x = sp * (q + r / 2.0)
        y = sp * (np.sqrt(3) / 2.0) * r
        initial_params.extend([x, y, 0.0])
    
    initial_params = np.array(initial_params)
    
    best_params = initial_params.copy()
    best_score = objective(initial_params, 0.0)
    
    # Try different outer angles
    best_overall_score = best_score
    best_overall_params = initial_params.copy()
    best_overall_angle = 0.0
    
    for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        try:
            result = optimize_layout(initial_params, outer_angle, max_iter=1000)
            if result.fun < best_overall_score:
                best_overall_score = result.fun
                best_overall_params = result.x.copy()
                best_overall_angle = outer_angle
        except Exception:
            pass
    
    # Final result
    inner = []
    for i in range(N):
        x = best_overall_params[3*i]
        y = best_overall_params[3*i+1]
        ang = best_overall_params[3*i+2]
        inner.append([float(x), float(y), float(ang)])
    
    # The outer hexagon is centered at origin with the optimal angle
    outer_cx, outer_cy = 0.0, 0.0
    outer_side = float(best_overall_score)
    outer_angle = float(best_overall_angle)
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle

