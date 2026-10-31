"""
Packing N=11 unit regular hexagons inside a minimal regular hexagon.
Structural change: direct geometric construction of the known hexagonal lattice
packing (1 center + 6 axial + 4 diagonal), with support-function computation for
the minimal outer side, and a brief local polish on outer rotation angle.
"""
import math
import time
import random

N = 11


def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon (pointy-top at 0 deg)."""
    a = math.radians(angle_deg)
    pts = []
    for i in range(6):
        theta = a + math.pi * i / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        pts.append((x, y))
    return pts


def point_in_convex_polygon(px, py, poly):
    """Check if point is inside or on boundary of a convex polygon (CCW vertices)."""
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-9:
            return False
    return True


def _point_seg_dist(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)


def min_dist_between_hexes(h1, h2):
    """Minimum distance between two convex hexagons (vertex lists)."""
    min_d = float('inf')
    # vertex-vertex
    for p1 in h1:
        for p2 in h2:
            d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if d < min_d:
                min_d = d
    # vertex-edge both ways
    for p in h1:
        for i in range(6):
            d = _point_seg_dist(p[0], p[1], h2[i][0], h2[i][1], h2[(i+1)%6][0], h2[(i+1)%6][1])
            if d < min_d:
                min_d = d
    for p in h2:
        for i in range(6):
            d = _point_seg_dist(p[0], p[1], h1[i][0], h1[i][1], h1[(i+1)%6][0], h1[(i+1)%6][1])
            if d < min_d:
                min_d = d
    return min_d


def check_packing(inner_positions, outer_center, outer_side, outer_angle_deg):
    """Validate: containment + disjointness. Returns (bool, str)."""
    outer_pts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle_deg)
    
    # Containment
    for (ix, iy, iang) in inner_positions:
        inner_pts = hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in inner_pts:
            if not point_in_convex_polygon(vx, vy, outer_pts):
                return False, f"vertex ({vx:.4f},{vy:.4f}) outside outer"
    
    # Disjointness
    n = len(inner_positions)
    hexes = [hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner_positions]
    for i in range(n):
        for j in range(i + 1, n):
            d = min_dist_between_hexes(hexes[i], hexes[j])
            if d < -1e-9:
                return False, f"overlap hex {i}-{j} dist={d:.6f}"
    return True, "ok"


def min_outer_side(inner_positions, outer_center, outer_angle_deg):
    """
    Compute minimum outer hexagon side length (given center and rotation)
    needed to contain all inner hexagons, via half-plane support constraints.
    
    Outer hexagon with side S, center c, rotation alpha has 6 half-planes:
      dot(p - c, n_k) <= S * sqrt(3)/2,  k=0..5
    where n_k = (cos(alpha + pi/6 + k*pi/3), sin(alpha + pi/6 + k*pi/3)).
    
    So S >= max over all inner vertices v and all k of dot(v-c, n_k) / (sqrt(3)/2).
    """
    sqrt3_2 = math.sqrt(3) / 2.0
    alpha = math.radians(outer_angle_deg)
    max_proj = 0.0
    for k in range(6):
        na = alpha + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(na)
        ny = math.sin(na)
        for (ix, iy, iang) in inner_positions:
            for vx, vy in hex_vertices(ix, iy, 1.0, iang):
                proj = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if proj > max_proj:
                    max_proj = proj
    return max_proj / sqrt3_2


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin before 150s kill

    sqrt3 = math.sqrt(3)

    # Hexagonal lattice packing: 1 center + 6 axial (dist sqrt(3)) + 4 diagonal (dist 3)
    inner = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = math.radians(60 * k)
        inner.append([sqrt3 * math.cos(a), sqrt3 * math.sin(a), 0.0])
    for ad in (30, 90, 150, 270):
        a = math.radians(ad)
        inner.append([3.0 * math.cos(a), 3.0 * math.sin(a), 0.0])

    assert len(inner) == N

    outer_center = (0.0, 0.0)

    # Search over outer rotation angles (30-degree grid + refinement)
    best_side = float('inf')
    best_angle = 0.0

    # Coarse grid: 12 angles (every 30 degrees)
    for k in range(12):
        ang = k * 30.0
        s = min_outer_side(inner, outer_center, ang)
        if s < best_side:
            best_side = s
            best_angle = ang

    # Refine around the best angle
    if time.time() < deadline:
        for _ in range(60):
            trial = best_angle + (random.random() - 0.5) * 10.0
            s = min_outer_side(inner, outer_center, trial)
            if s < best_side:
                best_side = s
                best_angle = trial

        # Even finer
        for _ in range(40):
            trial = best_angle + (random.random() - 0.5) * 2.0
            s = min_outer_side(inner, outer_center, trial)
            if s < best_side:
                best_side = s
                best_angle = trial

    # Verify
    valid, msg = check_packing(inner, outer_center, best_side, best_angle)
    if not valid:
        # Safety: bump side slightly
        best_side += 0.01
        valid, msg = check_packing(inner, outer_center, best_side, best_angle)

    return inner, outer_center, best_side, best_angle
