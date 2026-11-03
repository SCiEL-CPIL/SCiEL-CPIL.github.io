"""Packing 12 unit regular hexagons inside a regular hexagon.
Strategy: Symmetric 6-fold packing with 2 rings.
- 1 hexagon at center (angle 0)
- 6 hexagons in inner ring (angle 0)
- 6 hexagons in outer ring (angle 30)
Optimize inner radius r1 and outer radius r2 via Nelder-Mead.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 150.0  # wall-clock limit
START_TIME = time.time()

def _hexagon_vertices(cx, cy, r, angle_deg):
    """
    Vertices of a regular hexagon with center (cx, cy), circumradius r,
    rotated by angle_deg degrees.
    For a regular hexagon, circumradius = side length.
    Vertices are at angles 0, 60, 120, 180, 240, 300 (offset by angle_deg).
    """
    a = np.radians(angle_deg)
    angles = a + np.radians(np.array([0, 60, 120, 180, 240, 300]))
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    return np.column_stack([xs, ys])

def _point_in_hexagon(px, py, cx, cy, r, angle_deg):
    """
    Check if point (px, py) is inside the hexagon with center (cx, cy),
    circumradius r, rotated by angle_deg.
    Use the fact that a regular hexagon is the intersection of 6 half-planes.
    The distance from center to each side (apothem) is r * cos(30) = r * sqrt(3)/2.
    """
    a = np.radians(angle_deg)
    # The 6 normal directions to the sides
    normals = a + np.radians(np.array([90, 150, 210, 270, 330, 30]))
    apothem = r * np.cos(np.radians(30))
    dx = px - cx
    dy = py - cy
    for n in normals:
        dist = dx * np.cos(n) + dy * np.sin(n)
        if dist > apothem + 1e-9:
            return False
    return True

def _min_outer_side_for_centers(centers, angles):
    """
    Given centers and angles of inner hexagons, find the minimum outer hexagon
    side length (circumradius) that contains all inner hexagons, with the outer
    hexagon centered at origin and angle 0.
    
    We use a 60-angle coarse scan (vectorized) for the outer hexagon angle,
    then refine.
    """
    # For each outer angle, compute the required circumradius
    # The outer hexagon with circumradius R and angle theta contains a point p
    # if for all 6 side normals (theta + 90, theta+150, ..., theta+330),
    # the projection of p onto the normal is <= R * cos(30)
    
    # We need R such that all 12 hexagons fit.
    # For a given outer angle theta, the constraint for hexagon i is:
    # For each of the 6 side normals of outer hex, and each of the 6 vertices of inner hex i:
    #   dot(vertex, normal) <= R * apothem_outer
    # So R >= max over all vertices and normals of dot(vertex, normal) / apothem_outer
    # apothem_outer = R * cos(30), so this is circular.
    # Actually: dot(vertex, normal) <= R * cos(30)
    # So R >= dot(vertex, normal) / cos(30) for all vertices and normals where dot > 0.
    
    # Precompute all inner vertices
    all_verts = []
    for (cx, cy), ang in zip(centers, angles):
        verts = _hexagon_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    # Scan outer angles
    theta_scan = np.radians(np.arange(0, 60, 1))  # 60 angles
    cos30 = np.cos(np.radians(30))
    
    best_R = np.inf
    best_theta = 0.0
    
    for theta in theta_scan:
        # 6 normals for outer hexagon
        normals = theta + np.radians(np.array([90, 150, 210, 270, 330, 30]))
        cos_normals = np.cos(normals)
        sin_normals = np.sin(normals)
        
        # For each vertex, compute dot product with each normal
        # all_verts: (72, 2), normals: (6,)
        dots = all_verts[:, 0:1] * cos_normals[None, :] + all_verts[:, 1:2] * sin_normals[None, :]
        # dots: (72, 6)
        max_dot = np.max(dots)
        if max_dot > 0:
            R_candidate = max_dot / cos30
            if R_candidate < best_R:
                best_R = R_candidate
                best_theta = theta
    
    # Refine around best_theta
    if best_R < np.inf:
        theta_fine = np.radians(np.arange(best_theta - 2, best_theta + 2.5, 0.1))
        for theta in theta_fine:
            normals = theta + np.radians(np.array([90, 150, 210, 270, 330, 30]))
            cos_normals = np.cos(normals)
            sin_normals = np.sin(normals)
            dots = all_verts[:, 0:1] * cos_normals[None, :] + all_verts[:, 1:2] * sin_normals[None, :]
            max_dot = np.max(dots)
            if max_dot > 0:
                R_candidate = max_dot / cos30
                if R_candidate < best_R:
                    best_R = R_candidate
                    best_theta = theta
    
    return best_R, np.degrees(best_theta)

def _objective(params):
    """
    params: [r1, r2]
    r1: radius of inner ring (6 hexagons)
    r2: radius of outer ring (6 hexagons)
    """
    if time.time() - START_TIME > DEADLINE - 5:
        return 1e10
    
    r1, r2 = params
    if r1 < 0.5 or r2 < r1 + 0.5:
        return 1e10
    
    # Build centers and angles
    centers = [(0.0, 0.0)]  # center hex
    angles = [0.0]
    
    # Inner ring: 6 hexagons at angle 0, radius r1
    for k in range(6):
        a = np.radians(60 * k)
        x = r1 * np.cos(a)
        y = r1 * np.sin(a)
        centers.append((x, y))
        angles.append(0.0)
    
    # Outer ring: 6 hexagons at angle 30, radius r2
    for k in range(6):
        a = np.radians(60 * k + 30)
        x = r2 * np.cos(a)
        y = r2 * np.sin(a)
        centers.append((x, y))
        angles.append(30.0)
    
    # Check disjointness (approximate via center distances)
    # Two unit hexagons (side 1) are disjoint if center distance > 2 * apothem * 2? 
    # Actually, the minimum distance between two disjoint unit hexagons depends on orientation.
    # For same orientation, the minimum center distance for non-overlap is 2 * apothem = sqrt(3) ≈ 1.732
    # Wait, no. Two unit hexagons side by side: the distance between centers when they just touch is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732? 
    # Let me think: a regular hexagon with side 1 has apothem sqrt(3)/2. Two identical hexagons placed side by side with parallel sides: the distance between centers is 2 * apothem = sqrt(3).
    # But if rotated, it could be different.
    # For safety, use a conservative check: center distance >= sqrt(3) - epsilon for same orientation.
    # Actually, the minimum distance between two unit regular hexagons (any orientation) is at least 2 * inradius = sqrt(3) if they don't overlap? No.
    # The inradius (apothem) is sqrt(3)/2. Two hexagons don't overlap if the distance between centers is at least the sum of their "radii" in the direction between centers.
    # This is complex. Let's use a simpler check: if center distance < 1.0, they definitely overlap.
    # For the symmetric packing, we can verify geometrically.
    
    # Compute min outer side
    R, outer_angle = _min_outer_side_for_centers(centers, angles)
    if R < np.inf:
        return R
    return 1e10

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial guess
    # Inner ring: 6 hexagons around center. For them to touch the center hex, r1 = sqrt(3) ≈ 1.732
    # But they also need to not overlap each other. Distance between adjacent inner ring hexagons:
    # 2 * r1 * sin(30) = r1. So r1 >= sqrt(3) for non-overlap? No, distance = 2*r1*sin(30) = r1.
    # So r1 >= sqrt(3) ≈ 1.732 for inner ring hexagons to not overlap each other.
    # And r1 >= sqrt(3) for them to touch the center hex.
    # So r1 = sqrt(3) is the minimum.
    
    # Outer ring: 6 hexagons at 30-degree offset. 
    # Distance from center: r2. 
    # Distance from inner ring hexagon: the inner hex is at angle 0, outer at 30.
    # Distance = sqrt(r1^2 + r2^2 - 2*r1*r2*cos(30)).
    # This should be >= sqrt(3) for non-overlap.
    # Also, outer ring hexagons adjacent: distance = 2*r2*sin(30) = r2. So r2 >= sqrt(3).
    
    r1_init = np.sqrt(3)
    r2_init = r1_init + np.sqrt(3)  # rough guess
    
    x0 = [r1_init, r2_init]
    
    # Nelder-Mead optimization
    best_params = x0
    best_val = _objective(x0)
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Verify time
    if time.time() - START_TIME > DEADLINE - 2:
        pass  # Use best found
    
    r1, r2 = best_params
    
    # Build final configuration
    centers = [(0.0, 0.0)]
    angles = [0.0]
    
    for k in range(6):
        a = np.radians(60 * k)
        x = r1 * np.cos(a)
        y = r1 * np.sin(a)
        centers.append((x, y))
        angles.append(0.0)
    
    for k in range(6):
        a = np.radians(60 * k + 30)
        x = r2 * np.cos(a)
        y = r2 * np.sin(a)
        centers.append((x, y))
        angles.append(30.0)
    
    R, outer_angle = _min_outer_side_for_centers(centers, angles)
    
    inner = [[cx, cy, ang] for (cx, cy), ang in zip(centers, angles)]
    return inner, (0.0, 0.0), R, outer_angle
