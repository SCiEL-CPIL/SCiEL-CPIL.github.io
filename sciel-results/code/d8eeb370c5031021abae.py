import math
import time
import numpy as np

N = 12
_START = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, s, rot_deg):
    """Vertices of a regular hexagon (flat-top base) centered at (cx,cy),
    side s, rotated by rot_deg degrees. 6 vertices CCW."""
    a = math.radians(rot_deg)
    v = []
    for i in range(6):
        ang = a + i * math.pi / 3.0
        v.append((cx + s * math.cos(ang), cy + s * math.sin(ang)))
    return v

def _support(vx, vy, cos_a, sin_a):
    return max(vx[i] * cos_a + vy[i] * sin_a for i in range(6))

def _required_side(vertices, rot_deg):
    """Minimal outer side (centered at origin) with rotation rot_deg that
    contains all given vertices. Returns side length."""
    a = math.radians(rot_deg)
    cos6 = [math.cos(a + k * math.pi / 3.0) for k in range(6)]
    sin6 = [math.sin(a + k * math.pi / 3.0) for k in range(6)]
    h = []
    for k in range(6):
        h.append(max(vertices[i][0] * cos6[k] + vertices[i][1] * sin6[k]
                     for i in range(len(vertices))))
    return max(h)

def _inner_positions(rot_deg):
    """3x4 hex lattice of unit hexagons, axis-aligned (rot 0) relative to outer.
    Centers on triangular lattice with spacing 2 (hex side 1, adjacent hexes
    share edges so they are tangent, disjoint interiors)."""
    # Flat-top unit hex: width (horizontal) = 2, vertical spacing = sqrt(3)
    # Triangular lattice basis:
    #   e1 = (2, 0)
    #   e2 = (1, sqrt(3))
    sqrt3 = math.sqrt(3.0)
    centers = []
    # 3 rows, 4 columns
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (r % 2) * 1.0
            y = r * sqrt3
            centers.append((x, y))
    # Center the cluster at origin (subtract centroid)
    cx = sum(p[0] for p in centers) / len(centers)
    cy = sum(p[1] for p in centers) / len(centers)
    centers = [(p[0] - cx, p[1] - cy) for p in centers]
    # Rotate the whole cluster by rot_deg (inner hexagons rotate with it)
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    rotated = []
    for (x, y) in centers:
        rotated.append((x * ca - y * sa, x * sa + y * ca))
    return rotated

def _all_vertices(rot_deg):
    """All 72 vertices of the 12 inner hexagons, in world coords."""
    centers = _inner_positions(rot_deg)
    verts = []
    for (cx, cy) in centers:
        for i in range(6):
            ang = math.radians(rot_deg) + i * math.pi / 3.0
            verts.append((cx + math.cos(ang), cy + math.sin(ang)))
    return verts

def _score(rot_deg):
    verts = _all_vertices(rot_deg)
    return _required_side(verts, rot_deg)

def _check_disjoint(rot_deg, eps=1e-7):
    """Verify inner hexagons are pairwise disjoint (non-overlapping interiors).
    For a triangular lattice with spacing 2 and side 1, adjacent hexagons are
    tangent (share an edge) which is allowed (disjoint = no interior overlap).
    We just sanity-check by distance between centers >= 2 - eps."""
    centers = _inner_positions(rot_deg)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < 2.0 - 1e-4:
                return False
    return True

def construct():
    t0 = time.time()

    # 1) Global coarse scan over rotation angles (0 to 60 deg, by symmetry).
    best_side = float('inf')
    best_rot = 0.0
    best_center = (0.0, 0.0)

    # Coarse scan
    n_coarse = 240
    for k in range(n_coarse + 1):
        if time.time() - t0 > _DEADLINE - 5:
            break
        rot = 60.0 * k / n_coarse
        s = _score(rot)
        if s < best_side:
            best_side = s
            best_rot = rot

    # 2) Local refinement around the best few coarse candidates.
    # Find top 5 coarse candidates
    coarse = []
    n_coarse = 240
    for k in range(n_coarse + 1):
        rot = 60.0 * k / n_coarse
        s = _score(rot)
        coarse.append((s, rot))
    coarse.sort()
    top5 = [r for (_, r) in coarse[:5]]

    # Refine each with a fine scan then golden-section
    for r0 in top5:
        if time.time() - t0 > _DEADLINE - 5:
            break
        # Fine scan in a window of +/- 2 deg
        lo, hi = max(0.0, r0 - 2.0), min(60.0, r0 + 2.0)
        n_fine = 200
        best_local = (float('inf'), r0)
        for k in range(n_fine + 1):
            r = lo + (hi - lo) * k / n_fine
            s = _score(r)
            if s < best_local[0]:
                best_local = (s, r)
        # Golden-section on a small window around best_local[1]
        c = best_local[1]
        w = 0.1
        lo, hi = max(0.0, c - w), min(60.0, c + w)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(60):
            if time.time() - t0 > _DEADLINE - 5:
                break
            m1 = hi - gr * (hi - lo)
            m2 = lo + gr * (hi - lo)
            f1 = _score(m1)
            f2 = _score(m2)
            if f1 < f2:
                hi = m2
            else:
                lo = m1
        c = (lo + hi) / 2.0
        s = _score(c)
        if s < best_side:
            best_side = s
            best_rot = c

    # 3) Build the final answer.
    # Outer hexagon: center at origin (we centered the cluster), side = best_side,
    # rotation = best_rot.
    # Inner hexagons: positions from _inner_positions(best_rot), each rotated by
    # best_rot (axis-aligned relative to outer).
    centers = _inner_positions(best_rot)
    inner = []
    for (x, y) in centers:
        inner.append([x, y, best_rot])

    # Verify disjointness
    if not _check_disjoint(best_rot):
        # Fallback: use the trivially valid seed-like layout (shouldn't happen)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return inner, (0.0, 0.0), best_side, best_rot
