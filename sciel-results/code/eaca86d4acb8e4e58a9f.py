"""Hexagonal Lattice Basis Optimization for 12 unit hexagons in a regular hexagon."""
import time
import math
import numpy as np

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

# The 12 lattice sites forming a connected hexagonal cluster (1-6-5 radial pattern)
# Coordinates in axial lattice basis (u, v) where basis vectors are at 60 degrees
LATTICE_SITES = [
    (0, 0),
    (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
    (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1),
]

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _poly_area_sign(pts):
    """Signed area (positive if CCW)."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0

def _poly_area(pts):
    return abs(_poly_area_sign(pts))

def _edge_crosses_segment(p1, p2, a, b):
    """Check if segment p1-p2 crosses segment a-b (proper crossing)."""
    def orient(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else -1
    o1 = orient(p1, p2, a)
    o2 = orient(p1, p2, b)
    o3 = orient(a, b, p1)
    o4 = orient(a, b, p2)
    if o1 != o2 and o3 != o4:
        return True
    return False

def _segments_intersect(p1, p2, p3, p4):
    return _edge_crosses_segment(p1, p2, p3, p4)

def _hexagons_disjoint(v1, v2, tol=1e-9):
    """Check two hexagons (lists of 6 vertices) are disjoint (no overlap, edges may touch)."""
    # Check if any vertex of v1 is strictly inside v2 or vice versa
    for v in v1:
        if _point_strictly_in_hex(v, v2):
            return False
    for v in v2:
        if _point_strictly_in_hex(v, v1):
            return False
    # Check edge crossings
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    return True

def _point_strictly_in_hex(p, hexverts):
    """Strictly inside (not on boundary)."""
    # Use winding / sign consistency with strict inequalities
    n = len(hexverts)
    signs = []
    for i in range(n):
        a = hexverts[i]
        b = hexverts[(i+1) % n]
        val = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(val) < 1e-9:
            return False  # on edge, not strictly inside
        signs.append(1 if val > 0 else -1)
    # all same sign means inside
    if all(s == signs[0] for s in signs):
        return True
    return False

def _bounding_hexagon_side(inner_verts_all, angle_deg):
    """Compute minimal side length of a regular hexagon (given rotation) enclosing all points.
    Returns (center_x, center_y, side)."""
    # A regular hexagon with side s, center c, rotation a contains point p iff
    # for each of 6 directions (normals), the projection of (p - c) onto normal <= s * cos(30)
    # Actually distance from center to each side = s * cos(30 deg) = s * sqrt(3)/2
    # The 6 outward normals are at angles a + 30 + k*60
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + math.pi/6 + k * math.pi/3
        normals.append((math.cos(theta), math.sin(theta)))
    
    pts = [v for vs in inner_verts_all for v in vs]
    if not pts:
        return 0.0, 0.0, 0.0
    
    # For a given center, the required half-width in each normal direction is max projection
    # We need center such that max projection - min projection <= 2 * s * sqrt(3)/2 = s*sqrt(3)
    # To minimize s, we want to minimize max over normals of (range of projections)
    # This is a convex problem; but for a fixed set, the optimal center is such that
    # the bounding hexagon is "tight" in all 6 directions.
    
    # Simple approach: the minimal enclosing regular hexagon (fixed rotation) has center
    # at the point minimizing the max over 6 directions of (proj_max - proj_min).
    # We can solve via: for each direction d, range_d = max(p.d) - min(p.d).
    # s = max_d(range_d) / sqrt(3).
    # But center doesn't affect range! Range is translation-invariant.
    # So s is determined purely by the point set and rotation.
    # Center can be anywhere that centers the ranges; choose midpoint of projections.
    
    ranges = []
    for nx, ny in normals:
        projs = [p[0]*nx + p[1]*ny for p in pts]
        ranges.append(max(projs) - min(projs))
    max_range = max(ranges)
    s = max_range / math.sqrt(3)
    
    # Center: for each direction, midpoint of projections
    # Choose center so that for each normal, the midpoint aligns.
    # Use average of the per-direction midpoints.
    cx = 0.0
    cy = 0.0
    for nx, ny in normals:
        projs = [p[0]*nx + p[1]*ny for p in pts]
        mid = (max(projs) + min(projs)) / 2.0
        cx += mid * nx
        cy += mid * ny
    cx /= 6.0
    cy /= 6.0
    return cx, cy, s

def _objective(params):
    """params = [u1x, u1y, u2x, u2y, angle_deg]
    u1, u2 are lattice basis vectors. Returns (side_length, valid, inner, center, angle)."""
    u1x, u1y, u2x, u2y, angle_deg = params
    
    # Compute inner hexagon centers
    centers = []
    for (u, v) in LATTICE_SITES:
        cx = u * u1x + v * u2x
        cy = u * u1y + v * u2y
        centers.append((cx, cy))
    
    # All inner hexagons share the same rotation angle_deg
    side_inner = 1.0
    inner_verts = []
    for (cx, cy) in centers:
        inner_verts.append(_hex_vertices(cx, cy, side_inner, angle_deg))
    
    # Check disjointness
    for i in range(len(inner_verts)):
        for j in range(i+1, len(inner_verts)):
            if not _hexagons_disjoint(inner_verts[i], inner_verts[j]):
                return (float('inf'), False, None, None, angle_deg)
    
    # Compute bounding hexagon
    cx, cy, s = _bounding_hexagon_side(inner_verts, angle_deg)
    return (s, True, centers, (cx, cy), angle_deg)

def construct():
    start = time.time()
    
    # Initial guess: lattice basis vectors at 60 degrees with spacing ~2 (hexagons side 1,
    # distance between centers in hex tiling is 2*sqrt(3)/2*... actually for unit hexagons
    # touching, center distance = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ~ 1.732)
    # But we want them disjoint (may touch). Let's start with spacing = 2.0 (some gap).
    # Basis vectors: u1 = (d, 0), u2 = (d/2, d*sqrt(3)/2) where d = center spacing.
    d = 2.0
    u1 = (d, 0.0)
    u2 = (d/2.0, d * math.sqrt(3)/2.0)
    angle0 = 0.0
    
    best_params = [u1[0], u1[1], u2[0], u2[1], angle0]
    best_side, valid, best_centers, best_center, best_angle = _objective(best_params)
    if not valid:
        # increase spacing
        d = 2.5
        u1 = (d, 0.0)
        u2 = (d/2.0, d * math.sqrt(3)/2.0)
        best_params = [u1[0], u1[1], u2[0], u2[1], angle0]
        best_side, valid, best_centers, best_center, best_angle = _objective(best_params)
    
    # Nelder-Mead
    def nm_objective(p):
        s, v, c, cc, a = _objective(p)
        return s if v else (s if s != float('inf') else 1e6)
    
    # Simple Nelder-Mead implementation
    def nelder_mead(f, x0, max_iter=200, tol=1e-8, deadline_check=None):
        n = len(x0)
        # Initial simplex
        simplex = [list(x0)]
        for i in range(n):
            pt = list(x0)
            step = 0.1 * abs(pt[i]) + 0.1
            pt[i] += step
            simplex.append(pt)
        
        values = [f(pt) for pt in simplex]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        for it in range(max_iter):
            if deadline_check and deadline_check():
                break
            # Sort
            order = sorted(range(n+1), key=lambda i: values[i])
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            if abs(values[-1] - values[0]) < tol:
                break
            
            # Centroid of all but worst
            centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
            
            # Reflection
            worst = simplex[-1]
            xr = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
            fr = f(xr)
            
            if fr < values[0]:
                # Expansion
                xe = [centroid[j] + gamma * (centroid[j] - worst[j]) for j in range(n)]
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    values[-1] = fe
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                simplex[-1] = xr
                values[-1] = fr
            else:
                # Contraction
                xc = [centroid[j] + rho * (worst[j] - centroid[j]) for j in range(n)]
                fc = f(xc)
                if fc < values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        values[i] = f(simplex[i])
        
        # Return best
        order = sorted(range(n+1), key=lambda i: values[i])
        best_i = order[0]
        return simplex[best_i], values[best_i]
    
    def deadline_check():
        return (time.time() - start) > DEADLINE
    
    # Run Nelder-Mead
    if (time.time() - start) < DEADLINE:
        best_params, best_val = nelder_mead(nm_objective, best_params, max_iter=300, deadline_check=deadline_check)
        s, v, c, cc, a = _objective(best_params)
        if v and s < best_side:
            best_side = s
            best_centers = c
            best_center = cc
            best_angle = a
    
    # Second phase: refine with smaller steps if time remains
    if (time.time() - start) < DEADLINE - 5:
        def nm_objective2(p):
            s, v, c, cc, a = _objective(p)
            return s if v else 1e6
        best_params2, _ = nelder_mead(nm_objective2, best_params, max_iter=150, tol=1e-10, deadline_check=deadline_check)
        s, v, c, cc, a = _objective(best_params2)
        if v and s < best_side:
            best_side = s
            best_centers = c
            best_center = cc
            best_angle = a
    
    # Build inner list
    inner = [[cx, cy, best_angle] for (cx, cy) in best_centers]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}, Center: {center}, Angle: {angle}")
    print(f"Inner count: {len(inner)}")
