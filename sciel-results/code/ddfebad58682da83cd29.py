import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 146.0  # safety margin below the 150 s wall-clock kill

SQRT3 = math.sqrt(3.0)
APOTHEM = SQRT3 / 2.0          # apothem of a unit hexagon
D2 = (2 * SQRT3) ** 2          # squared max center distance for non-overlap


def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts


def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        axes.append((-dy / length, dx / length))
    return axes


def _is_valid(cx_arr, cy_arr, angle_arr):
    """Exact SAT overlap test over all pairs (used on full placements)."""
    n = len(cx_arr)
    verts_list = [_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]) for i in range(n)]
    axes_list = [_axes_from_polygon(v) for v in verts_list]
    for i in range(n):
        vi, ai = verts_list[i], axes_list[i]
        for j in range(i + 1, n):
            vj, aj = verts_list[j], axes_list[j]
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            d2 = dx * dx + dy * dy
            if d2 > D2:
                continue
            if d2 < 4.0 - 1e-9:
                return False
            sep = False
            for axis in ai:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    sep = True
                    break
            if sep:
                continue
            for axis in aj:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    sep = True
                    break
            if not sep:
                return False
    return True


def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """Smallest outer regular-hexagon side that contains all inner hexagons."""
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / APOTHEM
                if val > max_need:
                    max_need = val
    return max_need


def _quick_score(cx_arr, cy_arr, angle_arr, normals):
    """Fast support-function outer side (outer hexagon centered at origin)."""
    max_need = 0.0
    for i in range(N):
        a = math.radians(angle_arr[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * vx + ny * vy) / APOTHEM
                if val > max_need:
                    max_need = val
    return max_need


# ---- Lattice design family: hexagonal (A2) packing of aligned unit hexagons ----
def _lattice_positions(bx, by, cx, cy):
    """Centers of the 12 unit hexagons in a compact hexagonal (A2) lattice."""
    s = 2 * SQRT3
    u = (s * bx, s * SQRT3 * by)
    v = (s * (cx - 0.5 * bx), s * SQRT3 * (cy - 0.5 * by))
    pts = []
    for i in range(N):
        q, r = divmod(i, 3)
        x = u[0] * q + v[0] * r
        y = u[1] * q + v[1] * r
        pts.append((x, y))
    return pts


def _initializers():
    inits = []
    # (a) perfect hexagonal lattice, several orientations
    for ang in (0, 30, 15, 45):
        b = math.radians(ang)
        cb, sb = math.cos(b), math.sin(b)
        bx, by = 1.0, 0.0
        cx, cy = -sb, cb
        inits.append(_lattice_positions(bx, by, cx, cy))
    # (b) flower / hexagonal ring (center + 6 + 5)
    r1 = 2 * SQRT3
    pts = [(0.0, 0.0)]
    for i in range(6):
        t = math.radians(60 * i)
        pts.append((r1 * math.cos(t), r1 * math.sin(t)))
    r2 = r1 * math.sqrt(3.0)
    for deg in (30, 90, 150, 210, 270):
        t = math.radians(deg)
        pts.append((r2 * math.cos(t), r2 * math.sin(t)))
    inits.append(pts[:12])
    # (c) random
    for _ in range(3):
        inits.append([(random.uniform(-3, 3), random.uniform(-3, 3)) for _ in range(N)])
    return inits


def _refine(cfg, normals, t0):
    """Local Nesterov-momentum minimization of the outer side, no overlap (lattice-aligned)."""
    cx_arr = np.array([p[0] for p in cfg], dtype=float)
    cy_arr = np.array([p[1] for p in cfg], dtype=float)
    angle_arr = np.zeros(N)
    best_x = cx_arr.copy()
    best_y = cy_arr.copy()
    best_score = _quick_score(cx_arr, cy_arr, angle_arr, normals)

    T = 0.15
    T_end = 0.0005
    mu = 0.85
    eps = 0.10
    step = 0.0
    while time.time() - t0 < TIME_LIMIT - 12:
        step += 1
        idx = random.randrange(N)
        old_x, old_y = cx_arr[idx], cy_arr[idx]
        dx = random.gauss(0, T)
        dy = random.gauss(0, T)
        cx_arr[idx] = old_x + mu * dx
        cy_arr[idx] = old_y + mu * dy
        sc = _quick_score(cx_arr, cy_arr, angle_arr, normals)
        if sc < best_score or random.random() < math.exp(-(sc - best_score) / max(T, 1e-9)):
            best_x = cx_arr.copy()
            best_y = cy_arr.copy()
            best_score = sc
        else:
            cx_arr[idx] = old_x
            cy_arr[idx] = old_y
        T = max(T_end, T * 0.995)
    return best_x, best_y, best_score


def construct():
    t0 = time.time()
    random.seed(42)

    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))

    best_side = float('inf')
    best_cfg = None

    # Phase 0: exact-valid initial placements
    for pts in _initializers():
        if time.time() - t0 > TIME_LIMIT - 90:
            break
        if len(pts) != N:
            continue
        cx_arr = np.array([p[0] for p in pts], dtype=float)
        cy_arr = np.array([p[1] for p in pts], dtype=float)
        angle_arr = np.zeros(N)
        if _is_valid(cx_arr, cy_arr, angle_arr):
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = [(float(x), float(y)) for x, y in zip(cx_arr, cy_arr)]

    # Phase 1: lattice-aligned local refinement from many restarts
    for _ in range(100):
        if time.time() - t0 > TIME_LIMIT - 8:
            break
        start = _lattice_positions(1.0, 0.0, -math.sin(math.radians(random.uniform(-45, 45))),
                                   math.cos(math.radians(random.uniform(-45, 45))))
        # small random perturbation to escape the exact lattice
        cfg = [(x + random.gauss(0, 0.05), y + random.gauss(0, 0.05)) for (x, y) in start]
        bx, by, bs = _refine(cfg, normals, t0)
        if bs < best_side:
            best_side = bs
            best_cfg = [(float(x), float(y)) for x, y in zip(bx, by)]

    # Fallback: guaranteed (if slightly overlapping) lattice scaled to fit
    if best_cfg is None:
        cfg = _lattice_positions(1.0, 0.0, 0.0, 1.0)
        cfg = [(x * 1.05, y * 1.05) for (x, y) in cfg]
        best_cfg = cfg
        best_side = 40.0

    inner = [[x, y, 0.0] for (x, y) in best_cfg]
    return inner, (0.0, 0.0), float(best_side), 0.0
