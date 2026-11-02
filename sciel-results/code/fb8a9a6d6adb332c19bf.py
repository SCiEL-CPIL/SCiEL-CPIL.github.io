"""Hexagonal lattice packing with exact minimal enclosing hexagon via convex hull."""
import math

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (pointy-top at angle 0) centered at (cx,cy)."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + math.radians(60.0 * i)
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _min_enclosing_hex_side(hull_pts, angle_deg):
    """Minimal side of a regular hexagon (given orientation angle) enclosing hull_pts.
    A regular hexagon centered at origin with inradius r (side = r*2/sqrt(3)) contains
    point p iff for all 6 edge normals n_i, n_i . p <= r. So r >= max_i (n_i . p).
    We can also choose center; but for a lattice cluster the natural center is origin.
    To be safe, we center at the centroid of the hull and take r = max over vertices of
    max_i (n_i . (p - center))."""
    # center at centroid of hull
    cx = sum(p[0] for p in hull_pts) / len(hull_pts)
    cy = sum(p[1] for p in hull_pts) / len(hull_pts)
    a = math.radians(angle_deg)
    r = 0.0
    for i in range(6):
        # outward normal direction for edge i (flat-top hexagon, edge normals at 0,60,...)
        ni = math.radians(60.0 * i) + a
        nx, ny = math.cos(ni), math.sin(ni)
        for (px, py) in hull_pts:
            d = nx * (px - cx) + ny * (py - cy)
            if d > r:
                r = d
    # inradius r -> side = r * 2 / sqrt(3)
    side = r * 2.0 / math.sqrt(3.0)
    return side, (cx, cy)

def construct():
    k = 3  # hex cluster side (rows of 3,4,5,4,3 = 19 hexagons)
    spacing = math.sqrt(3.0)  # edge-sharing center distance for unit hexagons
    inner = []
    all_verts = []
    for r in range(-(k - 1), k):
        for q in range(-(k - 1), k):
            if abs(q + r) > k - 1:
                continue
            x = spacing * (q + r / 2.0)
            y = spacing * (math.sqrt(3.0) / 2.0) * r
            inner.append([x, y, 0.0])
            all_verts.extend(_hex_vertices(x, y, 1.0, 0.0))
    hull = _convex_hull(all_verts)
    side, (cx, cy) = _min_enclosing_hex_side(hull, 0.0)
    # add tiny epsilon to avoid boundary violations from float error
    side += 1e-9
    return inner, (cx, cy), side, 0.0
# EVOLVE-BLOCK-END
