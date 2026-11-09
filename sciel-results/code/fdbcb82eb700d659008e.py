# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg.
    Vertex i at angle a + 60*i degrees."""
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])


def _hex_normals(a_deg):
    """12 edge normals (unit vectors) for a hexagon of rotation a_deg.
    For each of 6 edges, the outward normal (pointing away from center) and its
    inward counterpart (toward center)."""
    a = np.radians(a_deg)
    # Edge i connects vertex i to vertex i+1. The outward normal bisects the
    # angles of those two vertices. Vertex angles: a + 60*i.
    # Outward normal angle for edge i = a + 60*i + 30 deg (bisector of vertex i and i+1)
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    # 12 normals: outward and inward for each edge
    return np.vstack([normals, -normals])


def _separation_ok(v1, v2, n1, n2):
    """Check if two hexagons (vertices v1, v2 with normals n1, n2) are separated.
    Separation: on at least one of the 12 edge normals of either hex, the
    projected intervals are apart by more than 1e-6."""
    # For each normal direction, project both hexagons and check gap
    # We check all 12 normals from hex1 and all 12 from hex2
    for normals in (n1, n2):
        for j in range(normals.shape[0]):
            n = normals[j]
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # Gap: max1 < min2 - 1e-6 or max2 < min1 - 1e-6
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
    return False


def _hexagons_separated(v1, v2, a1, a2):
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    return _separation_ok(v1, v2, n1, n2)


def _point_in_hexagon(p, outer_verts):
    """Check if point p is inside (or on boundary of) the outer hexagon.
    outer_verts: (6,2) array of vertices in order."""
    # For each edge from v_i to v_{i+1}, check cross product >= -1e-6
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        to_p = p - p1
        cross = edge[0] * to_p[1] - edge[1] * to_p[0]
        if cross < -1e-6:
            return False
    return True


def _all_inner_in_outer(inner_verts_list, outer_verts):
    """Check all vertices of all inner hexagons are inside the outer hexagon."""
    for verts in inner_verts_list:
        for p in verts:
            if not _point_in_hexagon(p, outer_verts):
                return False
    return True


def _build_hexagons(params):
    """
    params: array of 5 values:
      [r1, r2, theta1, theta2, outer_angle]
    r1: radius of inner ring (6 hexagons)
    r2: radius of outer ring (4 hexagons)
    theta1: angular offset for inner ring (deg)
    theta2: angular offset for outer ring (deg)
    outer_angle: rotation of outer hexagon (deg)
    
    Returns list of (cx, cy, angle_deg) for 11 hexagons.
    """
    r1, r2, theta1, theta2, outer_angle = params
    hexagons = []
    # Center hexagon
    hexagons.append((0.0, 0.0, 0.0))
    # Inner ring: 6 hexagons at radius r1, angles theta1 + 60*i
    for i in range(6):
        ang = np.radians(theta1 + 60.0 * i)
        cx = r1 * np.cos(ang)
        cy = r1 * np.sin(ang)
        hexagons.append((cx, cy, 0.0))
    # Outer ring: 4 hexagons at radius r2, angles theta2 + 90*i (or some pattern)
    # Actually, let's use 4 at 90-degree intervals for a more compact arrangement
    # Or maybe 4 at angles that fill gaps. Let me think about this differently.
    # 
    # Better: use a more standard hexagonal packing. 
    # For 11 hexagons: center + 6 + 4 doesn't form a nice pattern.
    # Let me use center + 6 + 4 where the 4 are at the "corners" of a larger hex.
    # Angles for outer 4: theta2 + 0, 90, 180, 270 degrees
    for i in range(4):
        ang = np.radians(theta2 + 90.0 * i)
        cx = r2 * np.cos(ang)
        cy = r2 * np.sin(ang)
        hexagons.append((cx, cy, 0.0))
    return hexagons


def _compute_outer_hex(hexagons):
    """Compute minimal enclosing regular hexagon for the given inner hexagons.
    Returns (center_x, center_y, angle_deg, side_length)."""
    # Get all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang) in hexagons:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # The outer hexagon must contain all these vertices.
    # For a regular hexagon with center (ox, oy), angle a, side s:
    # The distance from center to each edge (apothem) = s * sqrt(3)/2
    # For each of the 6 edge normals, the max projection of all points must be <= apothem
    # So s >= max_over_normals(max_projection - min_projection_centered) / sqrt(3)
    
    # We need to find the best (ox, oy, a) that minimizes s.
    # For a given angle a, the optimal center is the centroid that minimizes the
    # maximum projection range. This is a 2D problem.
    
    # Simplification: assume center is at origin (by symmetry of our packing).
    # Then for a given angle a, s = max_over_6_normals(max_proj - 0) / (sqrt(3)/2)
    # Wait, the projection must be within [-apothem, apothem] if centered at origin.
    # Actually, the hexagon centered at origin with angle a has edges at distance
    # apothem = s*sqrt(3)/2 from center. So for each normal direction n_i,
    # we need max(p @ n_i) <= apothem and min(p @ n_i) >= -apothem.
    # So apothem >= max_i(max(p @ n_i), -min(p @ n_i))
    # s = apothem / (sqrt(3)/2)
    
    # Let's optimize over angle a only, assuming center at origin.
    best_s = np.inf
    best_a = 0.0
    
    for a_deg in np.linspace(0, 60, 61):
        a = np.radians(a_deg)
        # 6 outward normals
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        proj = all_verts @ normals.T  # (N, 6)
        max_proj = proj.max(axis=0)  # (6,)
        min_proj = proj.min(axis=0)  # (6,)
        
        # Need apothem >= max(max_proj, -min_proj) for each normal
        required_apothem = np.maximum(max_proj, -min_proj)
        apothem = required_apothem.max()
        s = apothem / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    return (0.0, 0.0, best_a, best_s)


def _objective(params):
    """Objective: minimize outer hexagon side length, with penalty for violations."""
    hexagons = _build_hexagons(params)
    
    # Check separation between all pairs
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]
    
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hexagons_separated(verts_list[i], verts_list[j], hexagons[i][2], hexagons[j][2]):
                penalty += 100.0
    
    # Compute outer hexagon
    ox, oy, oa, s = _compute_outer_hex(hexagons)
    
    # Check containment
    outer_verts = _hex_vertices(ox, oy, s, oa)
    if not _all_inner_in_outer(verts_list, outer_verts):
        penalty += 100.0
    
    return s + penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Initial guess: r1 ~ 3 (for touching hexes in ring), r2 ~ 5
    # For unit hexagons, the distance between centers of touching hexagons is 2*apothem = sqrt(3) ≈ 1.732
    # Wait, no. For two unit hexagons to touch, their centers must be separated by 2 * (inradius) = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732? 
    # Actually, the inradius (apothem) of a unit hexagon is sqrt(3)/2. Two hexagons touch when the distance between centers equals 2 * apothem = sqrt(3).
    # But that's for parallel edges. The actual minimum distance for non-overlapping hexagons depends on orientation.
    # For two unit hexagons with the same orientation, they touch when center distance = sqrt(3) (edge-to-edge) or 2 (vertex-to-vertex along flat direction).
    # The minimum center distance for non-overlapping same-oriented unit hexagons is sqrt(3) ≈ 1.732.
    
    # So for a ring of 6 around center, r1 should be about sqrt(3) ≈ 1.732
    # For outer ring, r2 should be about 2*sqrt(3) ≈ 3.464
    
    initial = np.array([np.sqrt(3), 2 * np.sqrt(3), 0.0, 0.0, 0.0])
    
    best_result = None
    best_obj = np.inf
    
    # Try multiple starting points
    start_points = [
        [np.sqrt(3), 2 * np.sqrt(3), 0.0, 0.0, 0.0],
        [np.sqrt(3), 2 * np.sqrt(3), 30.0, 45.0, 0.0],
        [1.7, 3.4, 0.0, 0.0, 15.0],
        [1.8, 3.5, 15.0, 30.0, 0.0],
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(_objective, sp, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4,
                                   'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to initial
        best_result = initial
    
    # Build final hexagons
    hexagons = _build_hexagons(best_result)
    inner_hex_data = np.array(hexagons)
    
    # Compute outer hexagon
    ox, oy, oa, s = _compute_outer_hex(hexagons)
    outer_hex_data = np.array([ox, oy, oa])
    
    # Verify and adjust if needed
    # Check all constraints; if violated, increase s
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]
    
    # Check separation
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hexagons_separated(verts_list[i], verts_list[j], hexagons[i][2], hexagons[j][2]):
                # Increase s to fix
                s *= 1.1
                break
    
    # Check containment
    outer_verts = _hex_vertices(ox, oy, s, oa)
    for _ in range(10):  # Iterate to fix containment
        if _all_inner_in_outer(verts_list, outer_verts):
            break
        s *= 1.01
        outer_verts = _hex_vertices(ox, oy, s, oa)
    
    # Ensure s >= sqrt(11)
    s = max(s, np.sqrt(11.0))
    
    return inner_hex_data, outer_hex_data, float(s)


# EVOLVE-BLOCK-END
