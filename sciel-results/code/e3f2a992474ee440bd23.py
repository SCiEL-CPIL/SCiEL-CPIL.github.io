"""Hexagon packing: 12 unit hexagons inside a minimal regular hexagon."""
import time
import math
import numpy as np

N = 12
_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0

def _hex_verts(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k)
        x = cx + side * (math.cos(t) * ca - math.sin(t) * sa)
        y = cy + side * (math.sin(t) * ca + math.cos(t) * sa)
        out.append((x, y))
    return out

def _outer_verts(cx, cy, side, angle_deg):
    return _hex_verts(cx, cy, side, angle_deg)

def _min_side_for_points(pts, center, angle_deg):
    """Smallest outer regular-hex side (fixed center+rotation) containing pts."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    need = 0.0
    for (x, y) in pts:
        dx, dy = x - center[0], y - center[1]
        # rotate into outer frame
        u = dx * ca - dy * sa
        v = dx * sa + dy * ca
        m = max(abs(u), abs(v), abs((u - v) * 0.5))
        if m > need:
            need = m
    return 2.0 * need / math.sqrt(3.0)

def _all_inner_verts(positions):
    pts = []
    for (x, y, ang) in positions:
        pts.extend(_hex_verts(x, y, 1.0, ang))
    return pts

def _construct_base():
    # 2 rows of 6, alternating 0/90 deg, centered at origin.
    positions = []
    for i in range(6):
        x = -5.0 + i
        positions.append([x, 0.0, 0.0])
    for i in range(6):
        x = -5.0 + i
        positions.append([x, 1.0 + math.sqrt(3.0) / 2.0, 90.0])
    return positions

def _best_layout(positions):
    """Try to reduce outer side by small shifts; return (positions, center, side, angle)."""
    base_pos = [list(p) for p in positions]
    base_side = _min_side_for_points(_all_inner_verts(base_pos), (0.0, 0.0), 30.0)
    best = (base_pos, (0.0, 0.0), base_side, 30.0)
    if time.time() > _DEADLINE - _MARGIN:
        return best
    # small local search over block translation + outer rotation
    pts = _all_inner_verts(base_pos)
    cand_rot = [30.0]
    for dr in (-2.0, -1.0, 1.0, 2.0):
        cand_rot.append(30.0 + dr)
    for ang in cand_rot:
        # center at centroid of the block
        cx = sum(p[0] for p in base_pos) / len(base_pos)
        cy = sum(p[1] for p in base_pos) / len(base_pos)
        side = _min_side_for_points(pts, (cx, cy), ang)
        if side < best[2] - 1e-9:
            best = (base_pos, (cx, cy), side, ang)
    return best

# EVOLVE-BLOCK-START
def construct():
    try:
        positions = _construct_base()
        positions, center, side, angle = _best_layout(positions)
        inner = [[p[0], p[1], p[2]] for p in positions]
        return inner, (center[0], center[1]), side, angle
    except Exception:
        # guaranteed valid fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
