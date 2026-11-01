"""Packing 12 unit hexagons inside a regular hexagon using grid-based construction."""
import math
import numpy as np
import time

N = 12
DEADLINE = 120.0
START_TIME = time.time()

def _hex_verts(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side length, and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        verts.append([cx + side * math.cos(theta), cy + side * math.sin(theta)])
    return np.array(verts)

def _point_in_hexagon(p, center, side, angle_deg):
    """Check if point p is inside a regular hexagon."""
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    angle = math.radians(-angle_deg)
    rx = dx * math.cos(angle) - dy * math.sin(angle)
    ry = dx * math.sin(angle) + dy * math.cos(angle)
    
    apothem = side * math.sqrt(3) / 2
    for i in range(6):
        theta = i * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        if rx * nx + ry * ny > apothem + 1e-10:
            return False
    return True

def _hexagons_disjoint(h1, h2, tol=1e-9):
    """Check if two hexagons are disjoint using SAT."""
    v1 = _hex_verts(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_verts(h2[0], h2[1], 1.0, h2[2])
    
    all_edges = []
    for i in range(6):
        all_edges.append(v1[i] - v1[(i+1) % 6])
        all_edges.append(v2[i] - v2[(i+1) % 6])
    
    for edge in all_edges:
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    return False

def _all_disjoint(hexagons):
    """Check all pairs of hexagons are disjoint."""
    for i in range(len(hexagons)):
        for j in range(i+1, len(hexagons)):
            if not _hexagons_disjoint(hexagons[i], hexagons[j]):
                return False
    return True

def _min_enclosing_hexagon(hexagons, outer_angle=0.0):
    """Find minimum side length of outer hexagon (with given angle) that contains all inner hexagons."""
    all_verts = []
    for h in hexagons:
        verts = _hex_verts(h[0], h[1], 1.0, h[2])
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    sqrt3 = math.sqrt(3)
    best_s = float('inf')
    best_cx, best_cy = 0, 0
    
    n = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = n + i * math.pi / 3 + math.pi / 6
        normals.append((math.cos(theta), math.sin(theta)))
    
    ranges = []
    for nx, ny in normals:
        projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        ranges.append((projs.max(), projs.min()))
    
    s_candidate = max((r[0] - r[1]) / sqrt3 for r in ranges)
    
    if s_candidate < best_s:
        best_s = s_candidate
        A = np.array(normals)
        b = np.array([(r[0] + r[1]) / 2 for r in ranges])
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        best_cx, best_cy = c[0], c[1]
    
    return best_s, best_cx, best_cy

def _is_valid(hexagons, outer_center, outer_side, outer_angle, tol=1e-7):
    """Check if the packing is valid."""
    for h in hexagons:
        verts = _hex_verts(h[0], h[1], 1.0, h[2])
        for v in verts:
            if not _point_in_hexagon(v, outer_center, outer_side, outer_angle):
                return False
    if not _all_disjoint(hexagons):
        return False
    return True

def _construct_hexagonal_cluster():
    """
    Construct a proper hexagonal cluster using the hexagonal lattice.
    For unit hexagons in a flat-top orientation:
    - Lattice vectors: a1 = (1.5, sqrt(3)/2), a2 = (0, sqrt(3))
    - Adjacent centers are sqrt(3) apart
    
    We use axial coordinates (q, r) where position = q*a1 + r*a2
    """
    sqrt3 = math.sqrt(3)
    a1 = (1.5, sqrt3 / 2)
    a2 = (0.0, sqrt3)
    
    # Axial coordinates for a hexagonal cluster of 12 hexagons
    # Center: (0,0)
    # First ring (6 hexagons): (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Second ring - pick 5 that form a nice shape: (2,0), (2,-1), (1,-2), (0,-2), (-1,1)
    # Actually let's use: (2,0), (2,-1), (2,-2), (1,-2), (0,-2)
    
    axial_coords = [
        (0, 0),      # center
        (1, 0),      # first ring
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (2, 0),      # second ring
        (2, -1),
        (2, -2),
        (1, -2),
        (0, -2),
    ]
    
    hexagons = []
    for q, r in axial_coords:
        x = q * a1[0] + r * a2[0]
        y = q * a1[1] + r * a2[1]
        hexagons.append((x, y, 0.0))
    
    return hexagons

def _construct_linear_packing():
    """
    Construct a linear packing: 12 hexagons in a row.
    For flat-top hexagons, adjacent centers are 1.5 apart horizontally
    when they share a vertical edge... wait, that's not right.
    
    For flat-top hexagons sharing an edge, the center-to-center distance is sqrt(3).
    If we arrange them in a straight line along the x-axis, we need to check
    which edges they share.
    
    Actually, let's just place them along a hexagonal lattice line.
    Using axial coordinates with r=0: (0,0), (1,0), (2,0), ..., (11,0)
    """
    sqrt3 = math.sqrt(3)
    a1 = (1.5, sqrt3 / 2)
    a2 = (0.0, sqrt3)
    
    hexagons = []
    for q in range(12):
        x = q * a1[0]
        y = q * a1[1]
        hexagons.append((x, y, 0.0))
    
    return hexagons

def _try_packing(hexagons, outer_angle=0.0):
    """Try to fit hexagons in a minimal outer hexagon. Returns (side, center, angle) or None."""
    side, cx, cy = _min_enclosing_hexagon(hexagons, outer_angle)
    
    if _is_valid(hexagons, (cx, cy), side, outer_angle):
        return side, (cx, cy), outer_angle
    return None

def construct():
    """Main entry point: construct the packing."""
    best_result = None
    best_side = float('inf')
    
    # Try the hexagonal cluster first - this should be valid
    hexagons = _construct_hexagonal_cluster()
    
    for outer_angle in [0.0, 30.0]:
        result = _try_packing(hexagons, outer_angle)
        if result is not None:
            side, center, angle = result
            if side < best_side:
                best_side = side
                best_result = (hexagons, center, side, angle)
    
    # Also try linear packing as fallback
    hexagons_linear = _construct_linear_packing()
    for outer_angle in [0.0, 30.0]:
        result = _try_packing(hexagons_linear, outer_angle)
        if result is not None:
            side, center, angle = result
            if side < best_side:
                best_side = side
                best_result = (hexagons_linear, center, side, angle)
    
    # If no valid packing found, use a large fallback with well-separated hexagons
    if best_result is None:
        inner = []
        d = math.sqrt(3) + 0.01
        positions = [(0, 0)]
        for i in range(11):
            theta = i * math.pi / 3
            positions.append((d * math.cos(theta), d * math.sin(theta)))
        
        for px, py in positions:
            inner.append([px, py, 0.0])
        
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0
        
        return inner, center, side, angle
    
    hexagons, center, side, angle = best_result
    inner = [[h[0], h[1], h[2]] for h in hexagons]
    
    return inner, center, side, angle
