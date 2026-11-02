"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon with center (cx, cy), side length, and rotation."""
    angle = np.radians(angle_deg)
    # For a flat-top hexagon, vertices are at angles 0, 60, 120, 180, 240, 300
    # But we need to be consistent. Let's use pointy-top or flat-top?
    # The standard "regular hexagon" in packing problems often has flat top.
    # Vertices for flat-top hexagon centered at origin with side length s:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # Let's define vertices relative to center, then rotate and translate.
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + angle
    verts = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    verts[:, 0] += cx
    verts[:, 1] += cy
    return verts

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside or on the boundary of the hexagon."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle = -np.radians(angle_deg)
    # Rotate point back
    ca = np.cos(angle)
    sa = np.sin(angle)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    # For a flat-top hexagon with side length s centered at origin:
    # The hexagon is defined by |x| <= s, |y| <= s*sqrt(3)/2, and |sqrt(3)*x + y| <= s*sqrt(3)
    # More precisely, the half-widths in the three directions are:
    # Along x: s
    # Along y: s*sqrt(3)/2
    # Along (sqrt(3)*x + y) direction: s*sqrt(3)
    
    # Let's use the support function approach. A point is inside if it satisfies all 6 half-plane inequalities.
    # The 6 outward normals for a flat-top hexagon (side s) are at angles 0, 60, 120, 180, 240, 300.
    # The distance from center to each side (apothem) is s*sqrt(3)/2.
    # So the condition is: for each normal n_i, n_i . (p - c) <= apothem
    
    normals = np.column_stack([np.cos(np.deg2rad([0, 60, 120, 180, 240, 300])), 
                               np.sin(np.deg2rad([0, 60, 120, 180, 240, 300]))])
    apothem = side * np.sqrt(3) / 2
    
    dots = normals @ np.array([lx, ly])
    return np.all(dots <= apothem + 1e-9)

def _hexagon_fits_in_outer(inner_verts, outer_center, outer_side, outer_angle):
    """Check if all vertices of the inner hexagon are inside the outer hexagon."""
    ox, oy = outer_center
    for i in range(6):
        vx, vy = inner_verts[i]
        if not _point_in_hexagon(vx, vy, ox, oy, outer_side, outer_angle):
            return False
    return True

def _check_packing(inner, outer_center, outer_side, outer_angle):
    """
    Check if the packing is valid.
    inner: list of [x, y, angle_deg]
    outer_center: (cx, cy)
    outer_side: side length of outer hexagon
    outer_angle: rotation of outer hexagon
    Returns (is_valid, max_violation)
    """
    n = len(inner)
    if n == 0:
        return True, 0.0
    
    # Get all inner hexagon vertices
    inner_verts_list = []
    for i in range(n):
        x, y, ang = inner[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
    
    # Check each inner hexagon is inside outer
    for i in range(n):
        if not _hexagon_fits_in_outer(inner_verts_list[i], outer_center, outer_side, outer_angle):
            return False, 1.0
    
    # Check pairwise non-overlap
    # Two convex polygons don't overlap if there exists a separating axis.
    # For two regular hexagons, we can check all 12 axes (6 from each)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(inner_verts_list[i], inner_verts_list[j]):
                return False, 1.0
    
    return True, 0.0

def _separated(poly1, poly2):
    """Check if two convex polygons are separated (non-overlapping) using SAT."""
    # Get all edges from both polygons
    for poly in [poly1, poly2]:
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            # Edge vector
            edge = p2 - p1
            # Normal (perpendicular)
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto the normal
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Check for separation
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
    return False

def _outer_hexagon_tight_fit(inner, outer_center, outer_angle):
    """
    Given inner hexagons, outer center, and outer angle, find the minimum side length
    such that all inner hexagons fit inside the outer hexagon.
    """
    n = len(inner)
    if n == 0:
        return 0.0
    
    # For a given outer hexagon (center, angle), the minimum side length is determined
    # by the maximum over all inner hexagon vertices of the "support function" value.
    # For a flat-top hexagon centered at (ox, oy) with angle theta, a point (px, py) is inside
    # if for all 6 directions k: n_k . ((px,py) - (ox,oy)) <= s * apothem_factor
    # where apothem = s * sqrt(3)/2
    
    ox, oy = outer_center
    angle = np.radians(outer_angle)
    normals = np.column_stack([np.cos(np.deg2rad([0, 60, 120, 180, 240, 300]) + angle), 
                               np.sin(np.deg2rad([0, 60, 120, 180, 240, 300]) + angle)])
    
    max_dot = 0.0
    for i in range(n):
        x, y, ang = inner[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            dots = normals @ np.array([dx, dy])
            max_dot = max(max_dot, np.max(dots))
    
    # apothem = s * sqrt(3)/2, so s = max_dot / (sqrt(3)/2) = 2 * max_dot / sqrt(3)
    return 2.0 * max_dot / np.sqrt(3)

def construct():
    """
    Use hexagonal lattice arrangement to pack unit hexagons.
    """
    N = 11
    
    # Hexagonal lattice arrangement
    # In a hexagonal lattice, the distance between adjacent hexagon centers is 2 * apothem = sqrt(3)
    # for unit hexagons (side 1). Wait, no.
    # For unit regular hexagons (side 1), the distance between centers of adjacent hexagons
    # in a honeycomb packing is 2 * (sqrt(3)/2) = sqrt(3) if they share an edge.
    # Actually, the distance between centers of two adjacent hexagons sharing an edge is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3).
    
    # Let's use a hexagonal grid. The basis vectors for a hexagonal lattice with spacing sqrt(3) are:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 3/2)
    # Wait, let me reconsider.
    
    # For regular hexagons of side 1, the center-to-center distance for edge-sharing is sqrt(3).
    # The hexagonal lattice can be defined with basis:
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 3/2)
    
    sqrt3 = np.sqrt(3)
    e1 = np.array([sqrt3, 0.0])
    e2 = np.array([sqrt3 / 2.0, 1.5])
    
    # Generate hexagonal lattice points in a hexagonal shape
    # We want a hexagonal arrangement of N hexagons.
    # A hexagonal number arrangement: for radius r, we have 3r(r+1)+1 points.
    # For r=1: 7 points, r=2: 19 points, r=3: 37 points.
    # N=11 is not a hexagonal number. Let's just use a hexagonal-shaped subset.
    
    # Let's generate all lattice points within a certain radius and pick N of them.
    # Or better, generate a hexagonal patch.
    
    # Let's try a different approach: use a hexagonal grid with indices (i, j)
    # where the position is i*e1 + j*e2.
    # We want to select N points that form a compact cluster.
    
    # Let's use a hexagonal neighborhood. Start from origin and add neighbors.
    # Actually, for simplicity, let's just use a rectangular subset of the hexagonal lattice
    # that forms a roughly hexagonal shape.
    
    # For N=11, let's use a 3x3 grid plus 2 extra, or a hexagonal patch.
    # Hexagonal patches:
    # r=0: 1 point
    # r=1: 7 points (1 + 6)
    # r=2: 19 points (7 + 12)
    # So for 11, we can use r=1 (7 points) plus 4 more from r=2.
    
    # Let's generate lattice points in order of distance from origin.
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            pos = i * e1 + j * e2
            dist = np.linalg.norm(pos)
            points.append((dist, i, j, pos))
    
    points.sort(key=lambda x: x[0])
    
    # Take the first N points
    selected = []
    for dist, i, j, pos in points:
        if len(selected) >= N:
            break
        # Check if this point is not already selected (by position)
        is_new = True
        for _, _, _, p in selected:
            if np.allclose(pos, p, atol=1e-6):
                is_new = False
                break
        if is_new:
            selected.append((dist, i, j, pos))
    
    # Create inner hexagons at these positions
    # All with angle 0 for simplicity
    inner = []
    for dist, i, j, pos in selected:
        inner.append([pos[0], pos[1], 0.0])
    
    # Now find the tight outer hexagon
    # Try different outer angles and centers
    best_side = float('inf')
    best_config = None
    
    # The outer hexagon should be centered at the centroid of the inner hexagons
    cx = np.mean([p[0] for p in inner])
    cy = np.mean([p[1] for p in inner])
    outer_center = (cx, cy)
    
    # Try different outer angles
    for outer_angle in range(0, 60, 5):
        side = _outer_hexagon_tight_fit(inner, outer_center, float(outer_angle))
        if side < best_side:
            best_side = side
            best_config = (inner, outer_center, side, float(outer_angle))
    
    # Also try centering at (0, 0) and other centers
    for test_cx in [0.0, cx]:
        for test_cy in [0.0, cy]:
            outer_center_test = (test_cx, test_cy)
            for outer_angle in range(0, 60, 5):
                side = _outer_hexagon_tight_fit(inner, outer_center_test, float(outer_angle))
                if side < best_side:
                    best_side = side
                    best_config = (inner, outer_center_test, side, float(outer_angle))
    
    if best_config is None:
        # Fallback
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_config
# EVOLVE-BLOCK-END
