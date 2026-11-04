import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _cluster_centers():
    """
    Return the 11 centers of the hexagonal cluster (1 center + 6 first ring + 4 second ring).
    These are fixed in the cluster's local coordinate frame.
    """
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        centers.append((x, y))
    return np.array(centers)

def _all_cluster_vertices(rotation_deg):
    """
    Compute all 66 vertices of the 11-hexagon cluster given a global rotation.
    Each inner hexagon is rotated by (rotation_deg + 30) so that its flat sides align
    with the hexagonal lattice directions.
    Returns array of shape (66, 2).
    """
    centers = _cluster_centers()
    # Rotate centers by rotation_deg
    theta = np.deg2rad(rotation_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    rotated_centers = centers @ rot_matrix.T
    
    # Each hexagon has vertices at angles 30, 90, 150, 210, 270, 330 (flat-top)
    # Relative to the hexagon center, rotated by rotation_deg
    hex_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + rotation_deg)
    hex_verts_local = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # For each center, add the 6 local vertices
    all_verts = []
    for i in range(11):
        verts = rotated_centers[i] + hex_verts_local
        all_verts.append(verts)
    return np.vstack(all_verts)

def _min_side_for_rotation(rotation_deg, outer_angle_deg=30.0):
    """
    Compute the minimum outer hexagon side length that contains all cluster vertices
    given a cluster rotation and outer hexagon orientation.
    
    The outer hexagon is centered at origin with side length S and orientation outer_angle_deg.
    A point (x,y) is inside the outer hexagon if and only if for each of the 6 edge normals,
    the dot product with the normal is <= S * cos(30°) = S * sqrt(3)/2.
    
    Actually, for a regular hexagon with side S centered at origin with a vertex at angle α,
    the distance from center to each edge (apothem) is S * sqrt(3)/2.
    The 6 edge normals are at angles α + 30°, α + 90°, α + 150°, α + 210°, α + 270°, α + 330°.
    
    So the constraint is: for each normal direction n_k, max over all vertices of (v · n_k) <= S * sqrt(3)/2
    and min over all vertices of (v · n_k) >= -S * sqrt(3)/2
    
    Therefore S >= max over all k and all vertices of |v · n_k| / (sqrt(3)/2)
    """
    verts = _all_cluster_vertices(rotation_deg)
    
    # Outer hexagon orientation: vertex at angle outer_angle_deg
    # Edge normals at outer_angle_deg + 30° + k*60° for k=0..5
    normal_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + outer_angle_deg + 30)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each vertex, compute dot products with all 6 normals
    dots = verts @ normals.T  # shape (66, 6)
    
    # The required S is max over all dots of |dot| / (sqrt(3)/2)
    max_abs_dot = np.max(np.abs(dots))
    side = max_abs_dot / (np.sqrt(3) / 2)
    
    return side

def _objective_rotation(theta_deg, outer_angle_deg=30.0):
    """Objective: outer side length for given cluster rotation."""
    return _min_side_for_rotation(theta_deg, outer_angle_deg)

def _verify_solution(inner, center, side, angle_deg, tol=1e-9):
    """Verify that all inner hexagons are inside the outer hexagon and non-overlapping."""
    outer_v = _outer_hex_vertices(center[0], center[1], side, angle_deg)
    
    # Check containment
    for x, y, a in inner:
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    # Check non-overlap (simple distance check between centers)
    # For unit hexagons on hexagonal lattice, minimum center distance is sqrt(3)
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            d = np.hypot(inner[i][0] - inner[j][0], inner[i][1] - inner[j][1])
            if d < np.sqrt(3) - 1e-6:
                return False
    
    return True

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # Fixed cluster geometry
    centers_local = _cluster_centers()
    
    # Try different outer hexagon orientations
    best_side = float('inf')
    best_rotation = 0.0
    best_outer_angle = 30.0
    
    # The outer hexagon orientation matters. Try several.
    # Due to symmetry, we only need to consider 0 to 30 degrees for the outer hexagon.
    outer_angles_to_try = [0.0, 15.0, 30.0]
    
    for outer_angle in outer_angles_to_try:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        # Coarse scan over rotation
        coarse_best = float('inf')
        coarse_best_rot = 0.0
        for rot in np.linspace(0, 30, 61):  # 30-degree range due to 6-fold symmetry
            s = _min_side_for_rotation(rot, outer_angle)
            if s < coarse_best:
                coarse_best = s
                coarse_best_rot = rot
        
        # Fine scan around the best
        fine_best = float('inf')
        fine_best_rot = coarse_best_rot
        for rot in np.linspace(coarse_best_rot - 2, coarse_best_rot + 2, 201):
            s = _min_side_for_rotation(rot, outer_angle)
            if s < fine_best:
                fine_best = s
                fine_best_rot = rot
        
        # Use scipy minimize_scalar for high precision
        try:
            result = minimize_scalar(
                _objective_rotation,
                args=(outer_angle,),
                bounds=(fine_best_rot - 1, fine_best_rot + 1),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 1000}
            )
            if result.success and result.fun < fine_best:
                fine_best = result.fun
                fine_best_rot = result.x
        except Exception:
            pass
        
        if fine_best < best_side:
            best_side = fine_best
            best_rotation = fine_best_rot
            best_outer_angle = outer_angle
    
    # Build the final solution
    theta = np.deg2rad(best_rotation)
    c = np.cos(theta)
    s = np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    rotated_centers = centers_local @ rot_matrix.T
    
    inner = []
    for i in range(11):
        x = float(rotated_centers[i, 0])
        y = float(rotated_centers[i, 1])
        a = float(best_rotation + 30.0)
        inner.append([x, y, a])
    
    center = (0.0, 0.0)
    side = best_side
    angle = best_outer_angle
    
    # Verify
    if not _verify_solution(inner, center, side, angle):
        # Fallback to seed solution
        eps = 1e-9
        centers = []
        centers.append((0.0, 0.0))
        for k in range(6):
            ang = np.deg2rad(k * 60)
            x = (np.sqrt(3) + eps) * np.cos(ang)
            y = (np.sqrt(3) + eps) * np.sin(ang)
            centers.append((x, y))
        for k in range(4):
            ang = np.deg2rad(30 + k * 60)
            x = (3.0 + 2*eps) * np.cos(ang)
            y = (3.0 + 2*eps) * np.sin(ang)
            centers.append((x, y))
        
        inner = [[x, y, 30.0] for x, y in centers]
        max_x = (2 + eps) * np.sqrt(3)
        max_y = 4 + 2 * eps
        s_x = 2 * max_x / np.sqrt(3)
        s_y = max_y
        side = max(s_x, s_y) + 1e-9
        angle = 30.0
    
    return inner, center, side, angle
