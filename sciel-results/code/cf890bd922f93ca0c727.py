"""Packing 12 unit hexagons in a regular hexagon."""
import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hex_min_center_dist(delta_deg):
    """Minimum center distance for two unit hexagons with relative angle delta_deg."""
    delta = np.radians(delta_deg)
    us = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    h1 = np.max(np.cos(np.outer(us, np.arange(6) * np.pi / 3)), axis=1)
    h2_neg = np.max(np.cos(np.outer(us, (delta + np.arange(6) * np.pi / 3))), axis=1)
    return np.min(h1 + h2_neg)

def _build_dist_table():
    table = {}
    for d in range(0, 31):
        table[d] = _hex_min_center_dist(d)
    return table

def _get_min_dist(delta_deg, table):
    d = abs(delta_deg) % 60
    if d > 30:
        d = 60 - d
    d = int(round(d))
    return table.get(d, table[0])

def _point_in_hex(px, py, verts):
    for j in range(6):
        x1, y1 = verts[j]
        x2, y2 = verts[(j + 1) % 6]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-10:
            return False
    return True

def _feasible(inner, oc, side, oangle, table):
    cx, cy = oc
    outer_v = _hex_vertices(cx, cy, side, oangle)
    for i in range(len(inner)):
        ix, iy, ia = inner[i]
        iv = _hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in iv:
            if not _point_in_hex(vx, vy, outer_v):
                return False
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            dx = inner[j][0] - inner[i][0]
            dy = inner[j][1] - inner[i][1]
            dist = np.sqrt(dx * dx + dy * dy)
            md = _get_min_dist(inner[i][2] - inner[j][2], table)
            if dist < md - 1e-9:
                return False
    return True

def _make_config1():
    s3 = np.sqrt(3)
    hexes = []
    for row in range(3):
        for col in range(4):
            x = col * s3
            y = row * 1.5 * s3
            if row % 2 == 1:
                x += s3 / 2
            hexes.append([x, y, 0.0])
    return hexes[:12]

def _make_config2():
    s3 = np.sqrt(3)
    hexes = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        hexes.append([s3 * np.cos(a), s3 * np.sin(a), 0.0])
    for k in range(5):
        a = k * 2 * np.pi / 5 + np.pi / 6
        hexes.append([2 * s3 * np.cos(a), 2 * s3 * np.sin(a), 0.0])
    return hexes[:12]

def _make_config3():
    s3 = np.sqrt(3)
    hexes = []
    ao = np.radians(15)
    for row in range(3):
        for col in range(4):
            x = col * s3
            y = row * 1.5 * s3
            if row % 2 == 1:
                x += s3 / 2
            xr = x * np.cos(ao) - y * np.sin(ao)
            yr = x * np.sin(ao) + y * np.cos(ao)
            hexes.append([xr, yr, 15.0])
    return hexes[:12]

def _make_config4():
    s3 = np.sqrt(3)
    hexes = []
    rows = [2, 3, 3, 2, 2]
    y = 0
    for count in rows:
        x0 = -(count - 1) * s3 / 2
        for c in range(count):
            hexes.append([x0 + c * s3, y, 0.0])
        y += 1.5 * s3
    return hexes[:12]

def _make_config5():
    s3 = np.sqrt(3)
    hexes = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        hexes.append([s3 * np.cos(a), s3 * np.sin(a), 30.0])
    for k in range(5):
        a = k * 2 * np.pi / 5
        hexes.append([2 * s3 * np.cos(a), 2 * s3 * np.sin(a), 0.0])
    return hexes[:12]

def _local_opt(inner, oc, side, oangle, table, iters=300):
    best = (inner, oc, side, oangle)
    for _ in range(iters):
        improved = False
        for frac in [0.995, 0.99, 0.985, 0.98]:
            if _feasible(best[0], best[1], best[2] * frac, best[3], table):
                best = (best[0], best[1], best[2] * frac, best[3])
                improved = True
                break
        if not improved:
            for i in range(N):
                for dx in [-0.04, 0.04]:
                    for dy in [-0.04, 0.04]:
                        ni = [list(x) for x in best[0]]
                        ni[i][0] += dx
                        ni[i][1] += dy
                        if _feasible(ni, best[1], best[2], best[3], table):
                            for f in [0.995, 0.99]:
                                if _feasible(ni, best[1], best[2] * f, best[3], table):
                                    best = (ni, best[1], best[2] * f, best[3])
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 110

    table = _build_dist_table()
    configs = [_make_config1(), _make_config2(), _make_config3(), _make_config4(), _make_config5()]

    best = None
    best_side = float('inf')

    for cfg in configs:
        if time.time() > deadline - 25:
            break
        xs = [p[0] for p in cfg]
        ys = [p[1] for p in cfg]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_r = max(np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) + 1.0 for p in cfg)
        hi = 2 * max_r / np.sqrt(3) + 0.5
        lo = max_r
        oc = (cx, cy)
        oa = 0.0

        if not _feasible(cfg, oc, hi, oa, table):
            hi *= 1.5
            while not _feasible(cfg, oc, hi, oa, table) and hi < 50:
                hi *= 1.2
            if hi >= 50:
                continue

        for _ in range(40):
            if time.time() > deadline - 25:
                break
            mid = (lo + hi) / 2
            if _feasible(cfg, oc, mid, oa, table):
                hi = mid
            else:
                lo = mid

        if hi < best_side:
            best_side = hi
            if time.time() < deadline - 35:
                b = _local_opt(cfg, oc, hi, oa, table, iters=200)
                if b[2] < best_side:
                    best_side = b[2]
                    best = b
            else:
                best = (cfg, oc, hi, oa)

    if best is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner, oc, side, oa = best
    if time.time() < deadline - 5:
        for f in [0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993]:
            if _feasible(inner, oc, side * f, oa, table):
                side *= f
            else:
                break
    return (inner, oc, side, oa)

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
