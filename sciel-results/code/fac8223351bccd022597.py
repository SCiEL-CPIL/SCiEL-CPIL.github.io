"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side length and rotation."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = -np.radians(angle_deg)
    # Rotate point
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Hexagon with flat top: |lx| <= side, |ly| <= side*sqrt(3)/2, and corner constraints
    # For a regular hexagon with vertices at angles 0,60,120,180,240,300 (pointy top by default in our vertex gen)
    # Actually our hex_vertices generates pointy-top when angle=0
    # Let's use the half-plane method
    for i in range(6):
        theta1 = np.radians(angle_deg) + i * np.pi / 3
        theta2 = np.radians(angle_deg) + (i + 1) * np.pi / 3
        # Edge from vertex i to vertex i+1
        v1 = side * np.array([np.cos(theta1), np.sin(theta1)])
        v2 = side * np.array([np.cos(theta2), np.sin(theta2)])
        # Normal pointing inward
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        # Check if point is on the correct side
        # The center (0,0) should be on the inward side
        center_dot = np.dot(-v1, normal)
        point_dot = np.dot(np.array([lx, ly]) - v1, normal)
        if np.sign(center_dot) != np.sign(point_dot) and point_dot > 1e-9:
            return False
    return True

def _hexagon_contains_point(px, py, hex_center, hex_angle, hex_side):
    """Check if point is inside a hexagon."""
    cx, cy = hex_center
    # Transform to local coordinates
    dx = px - cx
    dy = py - cy
    angle = -np.radians(hex_angle)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with circumradius = hex_side (our convention)
    # The hexagon can be described by: max of |lx*cos(60k) + ly*sin(60k)| for k=0..5 <= hex_side * cos(30) = hex_side * sqrt(3)/2
    # Actually, the inradius is hex_side * sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes
    # For vertex at angle theta_k = hex_angle + k*60, the edge opposite has normal at theta_k
    # The distance from center to each edge is hex_side * sqrt(3)/2
    
    inradius = hex_side * np.sqrt(3) / 2
    for k in range(6):
        theta = np.radians(hex_angle) + k * np.pi / 3
        # Normal direction
        nx = np.cos(theta)
        ny = np.sin(theta)
        # Project local point onto normal
        proj = lx * nx + ly * ny
        if proj > inradius + 1e-12:
            return False
    return True

def _hexagon_min_dist(p, hex_center, hex_angle, hex_side):
    """Minimum distance from point p to hexagon (negative if inside)."""
    px, py = p
    cx, cy = hex_center
    dx = px - cx
    dy = py - cy
    angle = -np.radians(hex_angle)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    inradius = hex_side * np.sqrt(3) / 2
    
    # Find maximum over the 6 supporting functions
    max_proj = 0
    for k in range(6):
        theta = np.radians(hex_angle) + k * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        proj = lx * nx + ly * ny
        if proj > max_proj:
            max_proj = proj
    
    return max_proj - inradius

def _check_disjoint(inner_positions, inner_angles, inner_sides):
    """Check if all pairs of hexagons are disjoint (no overlap)."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            # Check if hexagons i and j overlap
            # Sample points on the boundary of hexagon i
            ci = np.array(inner_positions[i])
            cj = np.array(inner_positions[j])
            dist = np.linalg.norm(ci - cj)
            # If centers are far enough, they're definitely disjoint
            # Max extent of each hexagon from center is 1.0 (circumradius)
            if dist >= 2.0 + 1e-9:
                continue
            # Check more carefully: sample boundary of hex i
            overlap = False
            for k in range(12):
                theta = k * np.pi / 6
                # Point on boundary of hex i
                px = ci[0] + np.cos(theta) * np.cos(np.radians(inner_angles[i]))
                py = ci[1] + np.cos(theta) * np.sin(np.radians(inner_angles[i]))
                # Actually need to use the hexagon shape properly
                # Let's just check if center of j is inside i, or center of i is inside j, or boundaries intersect
                # Simple check: if center of i is inside hex j or center of j is inside hex i
                if _hexagon_contains_point(ci[0], ci[1], inner_positions[j], inner_angles[j], inner_sides[j]):
                    overlap = True
                    break
                if _hexagon_contains_point(cj[0], cj[1], inner_positions[i], inner_angles[i], inner_sides[i]):
                    overlap = True
                    break
            if not overlap:
                # Check edge-edge intersections by sampling
                for k in range(12):
                    theta = k * np.pi / 6
                    # Point on edge of hex i
                    # Get two adjacent vertices
                    t1 = np.radians(inner_angles[i]) + k * np.pi / 6
                    t2 = np.radians(inner_angles[i]) + (k + 1) * np.pi / 6
                    v1 = ci + np.array([np.cos(t1), np.sin(t1)])
                    v2 = ci + np.array([np.cos(t2), np.sin(t2)])
                    # Check if this edge segment intersects hex j
                    # Sample points along the edge
                    for frac in [0.0, 0.5, 1.0]:
                        px = v1[0] + frac * (v2[0] - v1[0])
                        py = v1[1] + frac * (v2[1] - v1[1])
                        if _hexagon_contains_point(px, py, inner_positions[j], inner_angles[j], inner_sides[j]):
                            overlap = True
                            break
                    if overlap:
                        break
            if overlap:
                return False
    return True

def _compute_outer_side(inner_positions, inner_angles, inner_sides, outer_center, outer_angle):
    """Compute the minimum outer hexagon side length that contains all inner hexagons."""
    oc = np.array(outer_center)
    oa = np.radians(outer_angle)
    
    # For each inner hexagon, find the maximum "radius" in the direction of each of the 6 outer hexagon normals
    # The outer hexagon with center oc, angle oa, side S contains a point p if for all k=0..5:
    #   (p - oc) . n_k <= S * sqrt(3)/2
    # where n_k is the unit normal at angle oa + k*60
    
    max_required = 0
    for i in range(len(inner_positions)):
        ci = np.array(inner_positions[i])
        # The inner hexagon's vertices
        for k in range(6):
            theta = np.radians(inner_angles[i]) + k * np.pi / 3
            v = ci + inner_sides[i] * np.array([np.cos(theta), np.sin(theta)])
            # Project onto each outer normal
            for m in range(6):
                nm = oa + m * np.pi / 3
                proj = np.dot(v - oc, np.array([np.cos(nm), np.sin(nm)]))
                if proj > max_required:
                    max_required = proj
    
    # Outer side S must satisfy S * sqrt(3)/2 >= max_required
    return max_required * 2 / np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # Initial arrangement: hexagonal lattice 4-3-4
    # Hexagonal lattice: spacing sqrt(3) between adjacent centers
    # Row spacing: 3/2 vertically
    s = np.sqrt(3)
    
    # 4-3-4 arrangement
    positions = []
    angles = [0.0] * 11
    
    # Row 0: 4 hexagons at y = 3
    for k in range(4):
        x = (k - 1.5) * s
        y = 3.0
        positions.append([x, y])
    
    # Row 1: 3 hexagons at y = 1.5
    for k in range(3):
        x = (k - 1.0) * s
        y = 1.5
        positions.append([x, y])
    
    # Row 2: 4 hexagons at y = 0
    for k in range(4):
        x = (k - 1.5) * s
        y = 0.0
        positions.append([x, y])
    
    positions = np.array(positions)
    
    # Center the arrangement
    center_offset = np.array([0.0, 1.5])
    positions = positions - center_offset
    
    # Compute initial outer hexagon
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    outer_side = _compute_outer_side(positions.tolist(), angles, [1.0]*11, outer_center, outer_angle)
    
    best_side = outer_side
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_outer_center = outer_center.copy()
    best_outer_angle = outer_angle
    
    # Optimize: try different outer hexagon angles and centers
    # The outer hexagon angle matters because the bounding hexagon shape changes
    # For a hexagonal lattice arrangement, rotating the outer hexagon by 30 degrees might help
    
    # Try various outer angles
    for oa_deg in range(0, 180, 15):
        oa = np.radians(oa_deg)
        # For each outer angle, find optimal center
        # The optimal center minimizes the max projection
        # This is a Chebyshev center problem in the hexagonal norm
        # Simple approach: try a grid of centers
        for cx in np.linspace(-2, 2, 9):
            for cy in np.linspace(-2, 2, 9):
                oc = [cx, cy]
                side = _compute_outer_side(positions.tolist(), angles, [1.0]*11, oc, oa_deg)
                if side < best_side:
                    best_side = side
                    best_outer_center = oc
                    best_outer_angle = oa_deg
    
    # Now try rotating the entire inner arrangement
    for rot_deg in range(0, 360, 15):
        rot = np.radians(rot_deg)
        cos_r = np.cos(rot)
        sin_r = np.sin(rot)
        rotated_positions = np.array([[p[0]*cos_r - p[1]*sin_r, p[0]*sin_r + p[1]*cos_r] for p in positions])
        
        for oa_deg in range(0, 180, 15):
            for cx in np.linspace(-2, 2, 7):
                for cy in np.linspace(-2, 2, 7):
                    oc = [cx, cy]
                    side = _compute_outer_side(rotated_positions.tolist(), angles, [1.0]*11, oc, oa_deg)
                    if side < best_side:
                        best_side = side
                        best_positions = rotated_positions.copy()
                        best_outer_center = oc
                        best_outer_angle = oa_deg
    
    # Also try a linear arrangement of 11 hexagons
    linear_positions = np.array([[k * s, 0.0] for k in range(11)])
    linear_positions = linear_positions - np.array([5.0 * s, 0.0])
    
    for oa_deg in range(0, 180, 15):
        for cx in np.linspace(-2, 2, 5):
            for cy in np.linspace(-2, 2, 5):
                oc = [cx, cy]
                side = _compute_outer_side(linear_positions.tolist(), angles, [1.0]*11, oc, oa_deg)
                if side < best_side:
                    best_side = side
                    best_positions = linear_positions.copy()
                    best_outer_center = oc
                    best_outer_angle = oa_deg
    
    # Verify disjointness of best arrangement
    if not _check_disjoint(best_positions.tolist(), best_angles, [1.0]*11):
        # Fallback to simple linear
        best_positions = linear_positions
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0
        best_side = _compute_outer_side(best_positions.tolist(), best_angles, [1.0]*11, best_outer_center, best_outer_angle)
    
    # Build inner list
    inner = []
    for i in range(11):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), float(best_angles[i])])
    
    return inner, (float(best_outer_center[0]), float(best_outer_center[1])), float(best_side), float(best_outer_angle)
# EVOLVE-BLOCK-END
