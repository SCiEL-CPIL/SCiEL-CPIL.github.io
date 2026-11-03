"""Fresh design: dense lattice seed + rigid-body settling + enclosing-hexagon refinement."""
import math
import time
import numpy as np

N = 12
_START = time.time()
_LIMIT = 150.0

def _hex_vertices(cx, cy, side, ang):
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    vx = [side * math.cos(math.radians(60 * i)) for i in range(6)]
    vy = [side * math.sin(math.radians(60 * i)) for i in range(6)]
    x = cx + ca * np.array(vx) - sa * np.array(vy)
    y = cy + sa * np.array(vx) + ca * np.array(vy)
    return np.column_stack((x, y))

def _max_support(verts, ang):
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    u = np.array([ca, sa])
    w = np.array([-sa, ca])
    pu = np.max(verts @ u)
    pw = np.max(verts @ w)
    pn = np.min(verts @ u)
    pm = np.min(verts @ w)
    d = 1.0 - 0.5 * (pu + pw)
    return max(pu, pw, -pn, -pm, d)

def _bounding_side(centers, angles, side, ang):
    allv = []
    for (cx, cy), aa in zip(centers, angles):
        allv.append(_hex_vertices(cx, cy, side, aa))
    allv = np.vstack(allv)
    return _max_support(allv, ang)

def _hexagon_overlap(A, B):
    for ang in (A[2], B[2]):
        u = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        w = np.array([-math.sin(math.radians(ang)), math.cos(math.radians(ang))])
        for d in (u, w, -u, -w):
            pa = np.max(A[0:2] @ d)
            pb = np.max(B[0:2] @ d)
            na = np.min(A[0:2] @ d)
            nb = np.min(B[0:2] @ d)
            if pb < na or pa < nb:
                return False
    return True

def _seed_positions():
    pos = []
    s = 1.0
    # compact 4x3 triangular lattice
    for j in range(-1, 2):
        for i in range(-1, 2):
            x = i * 2 * s + (0.5 if (j % 2) else 0.0)
            y = j * math.sqrt(3) * s
            pos.append((x, y, 0.0))
    # 4 more to reach 12, placed in gaps
    extra = [
        (1.5 * s, math.sqrt(3) * s, 0.0),
        (-1.5 * s, math.sqrt(3) * s, 0.0),
        (1.5 * s, -math.sqrt(3) * s, 0.0),
        (-1.5 * s, -math.sqrt(3) * s, 0.0),
    ]
    pos.extend(extra)
    return pos[:N]

def _settling(pos):
    # repulsive + containment settling
    pos = [list(p) for p in pos]
    lr = 0.02
    for it in range(400):
        if (time.time() - _START) > _LIMIT - 20:
            break
        for i in range(len(pos)):
            cx, cy, ang = pos[i]
            fx = fy = 0.0
            vi = _hex_vertices(cx, cy, 1.0, ang)
            for j in range(len(pos)):
                if i == j:
                    continue
                vj = _hex_vertices(pos[j][0], pos[j][1], 1.0, pos[j][2])
                if _hexagon_overlap(vi, vj):
                    dx = cx - pos[j][0]
                    dy = cy - pos[j][1]
                    L = math.hypot(dx, dy) + 1e-9
                    fx += 0.5 * dx / L
                    fy += 0.5 * dy / L
            # containment: keep within a soft circle
            r = math.hypot(cx, cy)
            if r > 3.0:
                fx -= 0.1 * cx / r
                fy -= 0.1 * cy / r
            pos[i][0] += lr * fx
            pos[i][1] += lr * fy
    return pos

def _refine_orientation(centers, angles, side):
    best_ang = 0.0
    best_val = _bounding_side(centers, angles, side, 0.0)
    # coarse
    for ang in range(0, 60, 3):
        v = _bounding_side(centers, angles, side, ang)
        if v < best_val:
            best_val = v
            best_ang = ang
    # fine
    for ang in np.arange(best_ang - 3, best_ang + 3, 0.25):
        v = _bounding_side(centers, angles, side, ang)
        if v < best_val:
            best_val = v
            best_ang = ang
    return best_ang, best_val

def construct():
    if (time.time() - _START) > _LIMIT - 10:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    pos = _settling(_seed_positions())
    centers = [(p[0], p[1]) for p in pos]
    angles = [p[2] for p in pos]

    best_ang, best_val = _refine_orientation(centers, angles, 1.0)

    # recenter on the bounding box centroid of all vertices
    allv = []
    for (cx, cy), aa in zip(centers, angles):
        allv.append(_hex_vertices(cx, cy, 1.0, aa))
    allv = np.vstack(allv)
    ccx = float((np.max(allv[:, 0]) + np.min(allv[:, 0])) / 2)
    ccy = float((np.max(allv[:, 1]) + np.min(allv[:, 1])) / 2)

    # shift inner so that outer center is at origin for reporting; keep relative geometry
    # We report outer center at (0,0) and shift inner by -c.
    inner = [[cx - ccx, cy - ccy, aa] for (cx, cy), aa in zip(centers, angles)]
    # recompute side with shifted centers (translation invariant)
    side = _bounding_side(inner, angles, 1.0, best_ang)

    return inner, (0.0, 0.0), float(side), float(best_ang)
