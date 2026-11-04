"""Packing 12 unit hexagons in a regular hexagon; minimize outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 12
DEADLINE = 140.0          # wall-clock safety margin (evaluator kills at 150s)
T0 = time.time()

# ---- geometry helpers ------------------------------------------------------
def _hex_verts(cx, cy, s, ang):
    a = math.radians(ang)
    c, sn = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k)
        x = s * math.cos(t); y = s * math.sin(t)
        out.append((cx + c * x - sn * y, cy + sn * x + c * y))
    return out

def _signed_area(poly):
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s

def _poly_intersect(a, b):
    # Sutherland-Hodgman: clip polygon 'a' by each edge of convex polygon 'b'
    def clip_edge(poly, p1, p2):
        def inside(p):
            return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) >= -1e-12
        def inter(p, q):
            ax, ay = p; bx, by = q
            x1, y1 = p1; x2, y2 = p2
            d = (x2 - x1) * (by - ay) - (y2 - y1) * (bx - ax)
            if abs(d) < 1e-12:
                return p
            t = ((x1 - ax) * (by - ay) - (y1 - ay) * (bx - ax)) / d
            return (ax + t * (bx - ax), ay + t * (by - ay))
        out = []
        if not poly:
            return out
        prev = poly[-1]
        for cur in poly:
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
            prev = cur
        return out
    poly = list(a)
    for i in range(len(b)):
        p1 = b[i]; p2 = b[(i + 1) % len(b)]
        poly = clip_edge(poly, p1, p2)
        if not poly:
            break
    return poly

def _overlap_area(a, b):
    inter = _poly_intersect(a, b)
    if len(inter) < 3:
        return 0.0
    return abs(_signed_area(inter))

def _min_dist(a, b):
    # min distance between two convex polygons (0 if overlapping)
    pts = list(a) + list(b)
    best = float('inf')
    for i in range(len(a)):
        for j in range(len(b)):
            d = (a[i][0] - b[j][0]) ** 2 + (a[i][1] - b[j][1]) ** 2
            if d < best:
                best = d
    return math.sqrt(best)

# ---- objective -------------------------------------------------------------
def _objective(params, outer_side, outer_ang):
    # params: [x0,y0,a0, x1,y1,a1, ...]
    inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
    ov = _hex_verts(0.0, 0.0, outer_side, outer_ang)
    pen = 0.0
    verts = []
    for p in inner:
        v = _hex_verts(p[0], p[1], 1.0, p[2])
        verts.append(v)
        a = _poly_intersect(v, ov)
        if len(a) >= 3:
            area = abs(_signed_area(a))
            pen += max(0.0, area - 1.0)          # part sticking outside
    for i in range(N):
        for j in range(i + 1, N):
            pen += _overlap_area(verts[i], verts[j])
    return pen

# ---- start layout ----------------------------------------------------------
def _initial_layout():
    # 3 rows x 4 cols, pointy-top unit hexes (flat-top orientation, angle 0)
    rows, cols = 3, 4
    dx = 3.0          # horizontal spacing (flat-top hex width = 2*side = 2, gap 1)
    dy = math.sqrt(3) # vertical spacing
    x0 = -1.5 * dx
    y0 = -dy
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append([x0 + c * dx, y0 + r * dy, 0.0])
    return out

def _initial_side(inner):
    # smallest outer side (flat-top, center 0) that covers the layout
    best = 2.0
    for p in inner:
        # max over 6 flat-top support directions of (v . n)
        for k in range(6):
            t = math.radians(60.0 * k)
            nx, ny = math.cos(t), math.sin(t)
            # support of unit flat-top hex in direction n
            sup = max(nx * math.cos(math.radians(60*m)) + ny * math.sin(math.radians(60*m)) for m in range(6))
            h = p[0]*nx + p[1]*ny + sup
            if h > best:
                best = h
    return best

# ---- simulated annealing ---------------------------------------------------
def _simulated_annealing(inner0, outer_side, outer_ang, budget):
    inner = [list(p) for p in inner0]
    cur_pen = _objective([x for p in inner for x in p], outer_side, outer_ang)
    best_inner = [list(p) for p in inner]
    best_pen = cur_pen
    T = 1.0
    alpha = 0.999
    rng = np.random.default_rng(0)
    while time.time() - T0 < budget:
        i = rng.integers(N)
        p = inner[i]
        cand = [p[0] + rng.normal(0, 0.4),
                p[1] + rng.normal(0, 0.4),
                (p[2] + rng.normal(0, 12.0)) % 60.0]
        inner[i] = cand
        new_pen = _objective([x for q in inner for x in q], outer_side, outer_ang)
        d = new_pen - cur_pen
        if d < 0 or rng.random() < math.exp(-d / max(T, 1e-9)):
            cur_pen = new_pen
            if new_pen < best_pen:
                best_pen = new_pen
                best_inner = [list(q) for q in inner]
        else:
            inner[i] = p
        T *= alpha
    return best_inner, best_pen

# ---- LBFGS polish (overlap-only, positions fixed outer) --------------------
def _lbfgs_polish(inner, outer_side, outer_ang, iters, ftol):
    x0 = np.array([x for p in inner for x in p])
    def f(x):
        return _objective(x, outer_side, outer_ang)
    res = minimize(f, x0, method='L-BFGS-B',
                   options={'maxiter': iters, 'ftol': ftol, 'gtol': 1e-8})
    x = res.x
    new_inner = [[x[3*i], x[3*i+1], x[3*i+2]] for i in range(N)]
    return new_inner, float(res.fun)

# ---- main construct --------------------------------------------------------
def construct():
    global T0
    T0 = time.time()
    outer_ang = 0.0
    inner = _initial_layout()
    outer_side = _initial_side(inner)

    # Phase 1: coarse annealing to reduce overlap
    budget1 = 35.0
    inner, pen = _simulated_annealing(inner, outer_side, outer_ang, budget1)

    # Phase 2: LBFGS polish (positions + angles)
    if time.time() - T0 < 95.0:
        inner, pen = _lbfgs_polish(inner, outer_side, outer_ang, iters=300, ftol=1e-12)

    # Phase 3: fine annealing around polished solution
    if time.time() - T0 < 110.0:
        inner, pen = _simulated_annealing(inner, outer_side, outer_ang, 15.0)

    # Phase 4: final LBFGS polish
    if time.time() - T0 < 130.0:
        inner, pen = _lbfgs_polish(inner, outer_side, outer_ang, iters=1500, ftol=1e-16)

    # Verify: ensure no overlap and all inside; if not, fall back to safe seed
    ov = _hex_verts(0.0, 0.0, outer_side, outer_ang)
    verts = [_hex_verts(p[0], p[1], 1.0, p[2]) for p in inner]
    valid = True
    for v in verts:
        a = _poly_intersect(v, ov)
        if len(a) < 3 or abs(_signed_area(a)) < 0.999:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                if _overlap_area(verts[i], verts[j]) > 1e-6:
                    valid = False
                    break
            if not valid:
                break

    if not valid:
        # safe fallback: original seed (guaranteed valid, side 40)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return inner, (0.0, 0.0), float(outer_side), float(outer_ang)
