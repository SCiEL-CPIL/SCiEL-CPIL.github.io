import math
import numpy as np

N = 11

def _cluster_points():
    # 3-5-3 hexagonal cluster on flat-top lattice
    pts = []
    # r = -1: q in -1..1
    for r in (-1, 0, 1):
        if r == -1:
            qs = (-1, 0, 1)
        elif r == 0:
            qs = (-2, -1, 0, 1, 2)
        else:
            qs = (-1, 0, 1)
        for q in qs:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
    return pts

def _hex_side_for_angle(pts, theta):
    # normals at theta + k*60
    widths = []
    for k in range(6):
        a = theta + k * math.pi / 3.0
        nx, ny = math.cos(a), math.sin(a)
        proj = [nx * x + ny * y for (x, y) in pts]
        widths.append(max(proj) - min(proj))
    return max(widths) / math.sqrt(3.0)

def _center_for_angle(pts, theta):
    mids = []
    for k in range(6):
        a = theta + k * math.pi / 3.0
        nx, ny = math.cos(a), math.sin(a)
        proj = [nx * x + ny * y for (x, y) in pts]
        m = (max(proj) + min(proj)) / 2.0
        mids.append((m * nx, m * ny))
    cx = sum(p[0] for p in mids) / 6.0
    cy = sum(p[1] for p in mids) / 6.0
    return (cx, cy)

def construct():
    import time
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin below 150s kill

    pts = _cluster_points()
    P = np.array(pts)

    # grid search over theta in [0, 60 deg)
    best_theta = 0.0
    best_L = float('inf')
    ngrid = 2000
    thetas = np.linspace(0.0, math.pi / 3.0, ngrid, endpoint=False)
    for th in thetas:
        L = _hex_side_for_angle(pts, float(th))
        if L < best_L:
            best_L = L
            best_theta = float(th)

    # local refinement (golden section) around best
    if time.time() < deadline:
        lo = best_theta - (math.pi / 3.0) / ngrid
        hi = best_theta + (math.pi / 3.0) / ngrid
        gr = (math.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = _hex_side_for_angle(pts, c)
        fd = _hex_side_for_angle(pts, d)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            if fc < fd:
                hi = d; d = c; fd = fc
                c = hi - gr * (hi - lo)
                fc = _hex_side_for_angle(pts, c)
            else:
                lo = c; c = d; fc = fd
                d = lo + gr * (hi - lo)
                fd = _hex_side_for_angle(pts, d)
        best_theta = (lo + hi) / 2.0
        best_L = _hex_side_for_angle(pts, best_theta)

    center = _center_for_angle(pts, best_theta)
    inner = [[x, y, 0.0] for (x, y) in pts]
    return inner, center, best_L, math.degrees(best_theta)
