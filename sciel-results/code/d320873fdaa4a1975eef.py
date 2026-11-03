"""Unit regular hexagons inside a regular hexagon (fresh design).

Strategy: place the N unit hexagons on a flat honeycomb lattice (the tightest
packing for equal hexagons), then find the smallest enclosing regular hexagon
by minimizing the support function of the point set over the 6 face normals,
scanning the rotation angle at the critical (vertex-aligned) angles.
"""
import math
import time

N = 11
T0 = time.time()
DEADLINE = 110.0  # safety margin before the 150 s wall-clock kill

def _hexagon_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon, circumradius s, flat-top at 0 deg."""
    a = math.radians(angle_deg)
    v = []
    for k in range(6):
        t = a + math.radians(60.0 * k)
        v.append((cx + s * math.cos(t), cy + s * math.sin(t)))
    return v

def _support(vertices, ux, uy):
    return max(x * ux + y * uy for x, y in vertices)

def _enclosing_side(vertices, angle_deg):
    """Smallest enclosing regular hexagon side length for a set of vertices,
    when the outer hexagon's face normals are at (angle_deg + 60k) degrees."""
    a = math.radians(angle_deg)
    mx = 0.0
    for k in range(6):
        t = a + math.radians(60.0 * k)
        mx = max(mx, _support(vertices, math.cos(t), math.sin(t)))
    return mx / math.cos(math.radians(30.0))

def _build_layouts():
    """Generate honeycomb point layouts (list of (x,y) centers)."""
    dirs = [(2.0, 0.0), (1.0, math.sqrt(3.0)), (-1.0, math.sqrt(3.0))]
    layouts = []

    # (a) Straight chain along one lattice direction.
    for d in range(3):
        pts = [(k * dirs[d][0], k * dirs[d][1]) for k in range(N)]
        layouts.append(pts)

    # (b) Compact triangular clusters (rows 1..2..3..2..1 = 11).
    row_counts = [1, 2, 3, 2, 1]
    for d in range(3):
        dx, dy = dirs[d]
        px, py = dirs[(d + 1) % 3]  # perpendicular-ish lattice step
        pts = []
        for i, c in enumerate(row_counts):
            for j in range(c):
                x = i * px + (j - (c - 1) / 2.0) * dx
                y = i * py + (j - (c - 1) / 2.0) * dy
                pts.append((x, y))
        layouts.append(pts)

    # (c) Staggered / brick-like rows (rows of 2,3,2,2,2 = 11) to compare.
    row_counts2 = [2, 3, 2, 2, 2]
    for d in range(3):
        dx, dy = dirs[d]
        px, py = dirs[(d + 2) % 3]
        pts = []
        for i, c in enumerate(row_counts2):
            for j in range(c):
                x = i * px + (j - (c - 1) / 2.0) * dx
                y = i * py + (j - (c - 1) / 2.0) * dy
                pts.append((x, y))
        layouts.append(pts)

    return layouts

def construct():
    best = None
    best_val = float("inf")

    for pts in _build_layouts():
        if time.time() > DEADLINE:
            break
        # All hexagons share the same rotation; use 0 for the inner shapes.
        verts = []
        for (x, y) in pts:
            verts.extend(_hexagon_vertices(x, y, 1.0, 0.0))

        # Scan the outer rotation at critical angles (30 deg resolution covers
        # all vertex/normal alignments for a flat-top point set).
        for adeg in range(0, 180, 30):
            val = _enclosing_side(verts, adeg)
            if val < best_val:
                best_val = val
                best = (pts, adeg)

        # Refine around the best coarse angle with a finer sweep.
        if best is not None and best[1] == adeg and time.time() < DEADLINE:
            for da in (-15, -10, -5, 5, 10, 15):
                av = adeg + da
                val = _enclosing_side(verts, av)
                if val < best_val:
                    best_val = val
                    best = (pts, av)

    if best is None:
        # Fallback: guaranteed-valid large container.
        pts = [(2.0 * k, 0.0) for k in range(N)]
        return [[x, y, 0.0] for (x, y) in pts], (0.0, 0.0), 40.0, 0.0

    pts, adeg = best
    inner = [[x, y, 0.0] for (x, y) in pts]
    return inner, (0.0, 0.0), best_val, adeg

# EVOLVE-BLOCK-START
# (construct and helpers above define the entry point)
# EVOLVE-BLOCK-END
