import time
import math
import numpy as np

N = 12
_EPS = 1e-9
_T0 = time.time()
_DEADLINE = 140.0  # safety margin before the 150s kill


def _hex_verts(cx, cy, s, a):
    th = np.deg2rad(a) + np.pi / 6.0
    ang = th + np.arange(6) * (np.pi / 3.0)
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _normals(cx, cy, s, a):
    th = np.deg2rad(a) + np.pi / 6.0
    ang = th + np.arange(6) * (np.pi / 3.0)
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _min_side(inner):
    """Minimal side of a regular hexagon (any center/rotation) enclosing all inner hexes."""
    n = len(inner)
    A = np.zeros((6 * n, 2))
    b = np.zeros(6 * n)
    for i in range(n):
        x, y, ang = inner[i][0], inner[i][1], inner[i][2]
        V = _hex_verts(x, y, 1.0, ang)
        Nn = _normals(x, y, 1.0, ang)
        A[6 * i:6 * i + 6] = Nn
        b[6 * i:6 * i + 6] = V @ Nn
    best = math.inf
    for i in range(6):
        for j in range(i + 1, 6):
            d1, d2 = A[i], A[j]
            det = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(det) < 1e-12:
                continue
            s = (b[i] * d2[1] - b[j] * d1[1]) / det
            if s < best - 1e-9:
                continue
            ok = True
            for k in range(6 * n):
                if A[k] @ d1 + b[k] * d2[1] - b[k] * d1[1] * 0.0:  # placeholder (unused)
                    pass
            for k in range(6 * n):
                # check A[k]·d1 + b[k]*d2[1] - b[k]*d1[1] ... use direct:
                val = A[k][0] * d1[0] + A[k][1] * d1[1] + b[k] * d2[1] - b[k] * d1[1]
                # correct check: A[k]·d1 + b[k]*d2[1] - b[k]*d1[1]  -> use:
                if A[k][0] * d1[0] + A[k][1] * d1[1] + b[k] * (d2[1] - d1[1]) < -_EPS:
                    ok = False
                    break
            if ok:
                best = s
    if best == math.inf:
        return None
    # recover center
    d1, d2 = A[i], A[j]
    det = d1[0] * d2[1] - d1[1] * d2[0]
    cx = (b[i] * d2[1] - b[j] * d2[0]) / det
    cy = (d1[0] * b[j] - d2[0] * b[i]) / det
    return best, (cx, cy), (math.degrees(math.atan2(d1[1], d1[0])) - 30.0) % 180.0


def _min_side_fast(inner):
    """Robust version of _min_side using support functions."""
    n = len(inner)
    # collect candidate directions
    dirs = []
    for i in range(n):
        x, y, ang = inner[i][0], inner[i][1], inner[i][2]
        th = np.deg2rad(ang) + np.pi / 6.0
        for k in range(6):
            a = th + k * (np.pi / 3.0)
            dirs.append((math.cos(a), math.sin(a)))
    best = math.inf
    best_info = None
    m = len(dirs)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = np.array(dirs[i])
            d2 = np.array(dirs[j])
            det = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(det) < 1e-12:
                continue
            # candidate side length from the two support constraints
            # For each inner hex h: max_v v·u = h_support(u)
            # outer: for all u, h_support(u) <= c·u + s
            # at u=d1: c·d1 = h1_max(d1) - s ; at u=d2: c·d2 = h2_max(d2) - s
            # where h1 is the hex attaining max at d1, h2 at d2
            # We need max over all hexes of support at d1 and d2
            h1 = 0.0
            for x, y, ang in inner:
                th = np.deg2rad(ang) + np.pi / 6.0
                sup = 0.0
                for k in range(6):
                    a = th + k * (np.pi / 3.0)
                    vx = x + math.cos(a)
                    vy = y + math.sin(a)
                    sup = max(sup, vx * d1[0] + vy * d1[1])
                h1 = max(h1, sup)
            h2 = 0.0
            for x, y, ang in inner:
                th = np.deg2rad(ang) + np.pi / 6.0
                sup = 0.0
                for k in range(6):
                    a = th + k * (np.pi / 3.0)
                    vx = x + math.cos(a)
                    vy = y + math.sin(a)
                    sup = max(sup, vx * d2[0] + vy * d2[1])
                h2 = max(h2, sup)
            s = (h1 * d2[1] - h2 * d1[1]) / det
            if s < best - 1e-9:
                continue
            # verify: for every direction u (use all dirs), support(u) <= c·u + s
            cx = (h1 * d2[1] - h2 * d2[0]) / det
            cy = (d1[0] * h2 - d2[0] * h1) / det
            ok = True
            for dx, dy in dirs:
                sup = 0.0
                for x, y, ang in inner:
                    th = np.deg2rad(ang) + np.pi / 6.0
                    for k in range(6):
                        a = th + k * (np.pi / 3.0)
                        vx = x + math.cos(a)
                        vy = y + math.sin(a)
                        sup = max(sup, vx * dx + vy * dy)
                if sup > cx * dx + cy * dy + s + 1e-7:
                    ok = False
                    break
            if ok:
                best = s
                best_info = (cx, cy, (math.degrees(math.atan2(d1[1], d1[0])) - 30.0) % 180.0)
    if best_info is None:
        return None
    return best, best_info


def _grid_positions():
    """Triangular lattice 4x3, side 1, touching."""
    dx = 3.0          # horizontal spacing between centers (2*R = 2 for touching? use 3 for gap)
    dy = math.sqrt(3) # vertical row spacing
    pts = []
    for row in range(3):
        off = 0.0 if row % 2 == 0 else 1.5
        for col in range(4):
            pts.append([off + col * 3.0, row * dy, 0.0])
    # center cluster
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    for p in pts:
        p[0] -= cx
        p[1] -= cy
    return pts


def construct():
    if time.time() > _DEADLINE:
        return _grid_positions(), (0.0, 0.0), 100.0, 0.0

    inner = _grid_positions()
    res = _min_side_fast(inner)
    if res is None:
        return inner, (0.0, 0.0), 100.0, 0.0
    side, (cx, cy), ang = res

    # Optional: small local polish of positions (only if time remains)
    if time.time() < _DEADLINE - 5.0:
        try:
            import random
            rng = random.Random(12345)
            best_side = side
            best_inner = [list(p) for p in inner]
            step = 0.05
            for it in range(300):
                if time.time() > _DEADLINE - 5.0:
                    break
                cand = [list(p) for p in best_inner]
                idx = rng.randrange(N)
                cand[idx][0] += rng.uniform(-step, step)
                cand[idx][1] += rng.uniform(-step, step)
                r = _min_side_fast(cand)
                if r is not None and r[0] < best_side - 1e-7:
                    best_side = r[0]
                    best_inner = cand
                    side, (cx, cy), ang = r
        except Exception:
            pass

    return best_inner, (cx, cy), best_side, ang


if __name__ == "__main__":
    print(construct())
