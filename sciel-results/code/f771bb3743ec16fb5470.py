"""Improved program for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
import math

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side 'side' and rotation angle_deg."""
    # Transform to local coordinates
    dx = px - cx
    dy = py - cy
    angle_rad = -math.radians(angle_deg)
    rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    
    # In local coords, hexagon with side s has vertices at angles 0, 60, ...
    # The hexagon can be described by |x| <= s and |y| <= s*sqrt(3)/2 and |x| + |y|/sqrt(3) <= s
    # More precisely, for a flat-top hexagon (angle=0), vertices are at (s,0), (s/2, s*sqrt(3)/2), etc.
    # Let's use the standard form: for a hexagon with "pointy top" vs "flat top"
    # With angle_deg=0, vertex 0 is at (s, 0). This is a flat-top orientation if we consider y-axis up.
    
    # A regular hexagon with circumradius R = side (since for regular hexagon, side = circumradius)
    # In local coords, the hexagon is the intersection of 6 half-planes.
    # For vertex at angle 0: the edge opposite is at x = -s
    # Actually, let's just check distance from center and angular position.
    
    # Simpler: use the fact that a regular hexagon with circumradius R can be defined as
    # the set of points where max over k of (dot product with normal_k) <= R * cos(pi/6)
    # Wait, that's for the inradius. Let me think again.
    
    # For a regular hexagon with side length s, the circumradius is s and inradius is s*sqrt(3)/2.
    # The hexagon is the intersection of 6 half-planes. The normals to the edges point at angles
    # 30, 90, 150, 210, 270, 330 degrees (for flat-top with vertex at 0).
    
    # Let's use a robust method: check if the point is inside by computing the maximum
    # of the dot products with the 6 edge normals.
    
    # Edge normals for a hexagon with vertex at angle 0 (in local coords):
    # The edges are perpendicular to directions 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    inradius = side * math.sqrt(3) / 2
    for nx, ny in normals:
        if rx * nx + ry * ny > inradius + 1e-9:
            return False
    return True

def hexagon_in_hexagon(inner_cx, inner_cy, inner_side, inner_angle,
                       outer_cx, outer_cy, outer_side, outer_angle):
    """Check if inner hexagon is completely inside outer hexagon."""
    # Check all vertices of inner hexagon are inside outer hexagon
    inner_verts = hex_vertices(inner_cx, inner_cy, inner_side, inner_angle)
    for vx, vy in inner_verts:
        if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def hexagons_disjoint(c1, c2, a1, a2, side1, side2, eps=1e-7):
    """Check if two hexagons are disjoint (non-overlapping)."""
    # For two convex polygons, they are disjoint if no vertex of one is inside the other
    # and no edges intersect. For simplicity, check vertices.
    # This is a sufficient but not necessary condition for disjointness in general,
    # but for convex shapes, if no vertex of A is in B and no vertex of B is in A,
    # and they don't share edges, they might still overlap. We need edge intersection check.
    
    # Let's use a simpler approach: check if any vertex of hex1 is inside hex2 or vice versa,
    # and also check for edge intersections.
    
    verts1 = hex_vertices(c1[0], c1[1], side1, a1)
    verts2 = hex_vertices(c2[0], c2[1], side2, a2)
    
    # Check if any vertex of 1 is inside 2
    for vx, vy in verts1:
        if point_in_hexagon(vx, vy, c2[0], c2[1], side2, a2):
            return False
    
    # Check if any vertex of 2 is inside 1
    for vx, vy in verts2:
        if point_in_hexagon(vx, vy, c1[0], c1[1], side1, a1):
            return False
    
    # Check for edge intersections
    def segments_intersect(p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects with p3-p4."""
        def orient(a, b, c):
            val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(val) < 1e-10:
                return 0
            return 1 if val > 0 else 2
        
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        
        if o1 != o2 and o3 != o4:
            return True
        
        # Collinear cases
        if o1 == 0 and on_segment(p1, p3, p2):
            return True
        if o2 == 0 and on_segment(p1, p4, p2):
            return True
        if o3 == 0 and on_segment(p3, p1, p4):
            return True
        if o4 == 0 and on_segment(p3, p2, p4):
            return True
        
        return False
    
    def on_segment(a, b, c):
        """Check if b is on segment a-c."""
        if (min(a[0], c[0]) - 1e-10 <= b[0] <= max(a[0], c[0]) + 1e-10 and
            min(a[1], c[1]) - 1e-10 <= b[1] <= max(a[1], c[1]) + 1e-10):
            return True
        return False
    
    for i in range(6):
        for j in range(6):
            p1 = verts1[i]
            p2 = verts1[(i + 1) % 6]
            p3 = verts2[j]
            p4 = verts2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    
    return True

def valid_configuration(positions, angles, outer_cx, outer_cy, outer_side, outer_angle, inner_side=1.0):
    """Check if a configuration is valid: all inner hexagons inside outer, and pairwise disjoint."""
    # Check all inside outer
    for i in range(len(positions)):
        cx, cy = positions[i]
        a = angles[i]
        if not hexagon_in_hexagon(cx, cy, inner_side, a, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    
    # Check pairwise disjoint
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if not hexagons_disjoint(positions[i], positions[j], angles[i], angles[j], inner_side, inner_side):
                return False
    
    return True

def compute_min_outer_side(positions, angles, outer_angle=0.0, outer_cx=0.0, outer_cy=0.0, inner_side=1.0):
    """Compute the minimum side length of an outer hexagon that contains all inner hexagons."""
    # For a given outer hexagon orientation, we need to find the minimum side length
    # such that all inner hexagon vertices are inside.
    
    # For a regular hexagon with center (cx, cy), side s, and angle theta,
    # a point (px, py) is inside if for all 6 edge normals n_k:
    #   dot((px-cx, py-cy), n_k) <= s * sqrt(3) / 2
    # where n_k are the unit normals to the edges.
    
    # So s >= max over all points and all k of dot((px-cx, py-cy), n_k) / (sqrt(3)/2)
    
    angle_rad = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = angle_rad + math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_val = 0.0
    for i in range(len(positions)):
        cx, cy = positions[i]
        a = angles[i]
        verts = hex_vertices(cx, cy, inner_side, a)
        for vx, vy in verts:
            dx = vx - outer_cx
            dy = vy - outer_cy
            for nx, ny in normals:
                val = dx * nx + dy * ny
                if val > max_val:
                    max_val = val
    
    if max_val <= 0:
        return 0.0
    
    return max_val / (math.sqrt(3) / 2)

def construct():
    """Construct a valid packing of 12 unit hexagons in a regular hexagon."""
    # Strategy: Use a grid-based approach with a single global rotation.
    # Place hexagons on a triangular lattice, then optimize the outer hexagon.
    
    # First, try a simple linear arrangement with optimization
    # Place 12 hexagons in a row, then find the best outer hexagon
    
    # Let's try several configurations:
    # 1. Linear row
    # 2. Two rows of 6
    # 3. Triangular arrangement
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    
    # Try different lattice arrangements
    # For a triangular lattice, the spacing between adjacent hexagon centers
    # must be at least 2*sqrt(3)/2 * side = sqrt(3) for side=1 (touching)
    # Actually, for two unit hexagons to be disjoint, their centers must be
    # at distance >= 2 * inradius = 2 * sqrt(3)/2 = sqrt(3) if aligned properly.
    # More precisely, the minimum distance between centers of two disjoint unit hexagons
    # depends on their relative orientation. For same orientation, it's sqrt(3).
    
    # Let's use a spacing of sqrt(3) + small epsilon for safety
    spacing = math.sqrt(3) + 0.01
    
    # Configuration 1: Single row of 12
    # Try different outer angles
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        positions = []
        angles = []
        for i in range(12):
            x = (i - 5.5) * spacing
            y = 0.0
            positions.append((x, y))
            angles.append(0.0)
        
        side = compute_min_outer_side(positions, angles, outer_angle, 0.0, 0.0)
        if side < best_side:
            best_side = side
            best_positions = positions
            best_angles = angles
            best_outer_angle = outer_angle
    
    # Configuration 2: Two rows of 6
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        positions = []
        angles = []
        for i in range(6):
            x = (i - 2.5) * spacing
            y = 0.0
            positions.append((x, y))
            angles.append(0.0)
        for i in range(6):
            x = (i - 2.5) * spacing
            y = spacing * math.sqrt(3) / 2
            positions.append((x, y))
            angles.append(0.0)
        
        side = compute_min_outer_side(positions, angles, outer_angle, 0.0, 0.0)
        if side < best_side:
            best_side = side
            best_positions = positions
            best_angles = angles
            best_outer_angle = outer_angle
    
    # Configuration 3: Triangular 4-4-4
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        positions = []
        angles = []
        # Row 0: 4 hexagons
        for i in range(4):
            x = (i - 1.5) * spacing
            y = 0.0
            positions.append((x, y))
            angles.append(0.0)
        # Row 1: 4 hexagons, offset
        for i in range(4):
            x = (i - 1.5) * spacing + spacing / 2
            y = spacing * math.sqrt(3) / 2
            positions.append((x, y))
            angles.append(0.0)
        # Row 2: 4 hexagons
        for i in range(4):
            x = (i - 1.5) * spacing
            y = spacing * math.sqrt(3)
            positions.append((x, y))
            angles.append(0.0)
        
        side = compute_min_outer_side(positions, angles, outer_angle, 0.0, 0.0)
        if side < best_side:
            best_side = side
            best_positions = positions
            best_angles = angles
            best_outer_angle = outer_angle
    
    # Configuration 4: 3x4 grid
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        positions = []
        angles = []
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * spacing
                y = row * spacing * math.sqrt(3) / 2
                positions.append((x, y))
                angles.append(0.0)
        
        side = compute_min_outer_side(positions, angles, outer_angle, 0.0, 0.0)
        if side < best_side:
            best_side = side
            best_positions = positions
            best_angles = angles
            best_outer_angle = outer_angle
    
    # Configuration 5: 4x3 grid
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        positions = []
        angles = []
        for row in range(4):
            for col in range(3):
                x = (col - 1.0) * spacing
                y = row * spacing * math.sqrt(3) / 2
                positions.append((x, y))
                angles.append(0.0)
        
        side = compute_min_outer_side(positions, angles, outer_angle, 0.0, 0.0)
        if side < best_side:
            best_side = side
            best_positions = positions
            best_angles = angles
            best_outer_angle = outer_angle
    
    # Now optimize the outer center and angle for the best configuration
    if best_positions is not None:
        # Optimize outer center
        def objective(params):
            cx, cy, angle = params
            return compute_min_outer_side(best_positions, best_angles, angle, cx, cy)
        
        # Initial guess
        x0 = [0.0, 0.0, best_outer_angle]
        
        # Use differential evolution for global optimization
        bounds = [(-10, 10), (-10, 10), (0, 180)]
        result = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-6)
        
        if result.fun < best_side:
            best_side = result.fun
            best_outer_cx = result.x[0]
            best_outer_cy = result.x[1]
            best_outer_angle = result.x[2]
        else:
            best_outer_cx = 0.0
            best_outer_cy = 0.0
    
    # Final result
    inner = []
    for pos, ang in zip(best_positions, best_angles):
        inner.append([pos[0], pos[1], ang])
    
    return inner, (best_outer_cx if best_positions else 0.0, best_outer_cy if best_positions else 0.0), best_side, best_outer_angle

# EVOLVE-BLOCK-END
