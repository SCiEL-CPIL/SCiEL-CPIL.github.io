import math
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
SQRT3 = math.sqrt(3)
EPS = 1e-9
BASE = np.array([0.0, SQRT3 / 2.0])
START = time.time()
DEADLINE = 130.0

# 6 support directions of a regular hexagon (normals to its sides), unit vectors
DIRS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

def _elapsed():
    return time.time() - START

def _hex_supports(cx, cy, angle_deg, s=1.0):
    """6 support values (dot with DIRS) of a unit hexagon at (cx,cy), rotated."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    local = s * (DIRS @ R.T)
    return local + np.array([cx, cy]) @ DIRS

def _outer_side(inner_params, outer_angle_deg, lo=1.0, hi=20.0):
    """Min outer side (via support functions) so all inner hexagons fit inside
    a regular hexagon centered at origin with the given orientation."""
    A = math.radians(outer_angle_deg)
    ca, sa = math.cos(A), math.sin(A)
    R = np.array([[ca, -sa], [sa, ca]])
    D = DIRS @ R.T
    # per-vertex support in outer frame: s_i = max_d DIRS_i . v
    S = np.zeros((N, 6))
    for i in range(N):
        x, y, ang = inner_params[i]
        a = math.radians(ang)
        ca2, sa2 = math.cos(a), math.sin(a)
        Rr = np.array([[ca2, -sa2], [sa2, ca2]])
        loc = DIRS @ Rr.T                       # 6x2
        S[i] = (loc * D).sum(axis=1) + (np.array([x, y]) @ D)
    M = S.max(axis=0)
    lo = max(lo, float(M.max()) + 1e-9)
    hi = max(hi, lo * 2 + 1.0)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if M.max() <= mid + 1e-9:
            hi = mid
        else:
            lo = mid
    return hi

def _separation(p1, p2):
    """Min distance between two convex hexagons (0 if overlapping/touching)."""
    # vertex containment
    for v in p1:
        if _point_in(v, p2):
            return 0.0
    for v in p2:
        if _point_in(v, p1):
            return 0.0
    best = float('inf')
    n1, n2 = 6, 6
    for i in range(n1):
        a1 = p1[i]; a2 = p1[(i + 1) % n1]
        for j in range(n2):
            b1 = p2[j]; b2 = p2[(j + 1) % n2]
            d = _segseg(a1, a2, b1, b2)
            if d < best:
                best = d
    return best

def _point_in(pt, poly):
    sign = 0
    for i in range(6):
        a = poly[i]; b = poly[(i + 1) % 6]
        c = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if c > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif c < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _segseg(p1, p2, p3, p4):
    d1 = p2 - p1; d2 = p4 - p3; r = p1 - p3
    a = np.dot(d1, d1); e = np.dot(d2, d2); f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = np.clip(f / e, 0, 1); return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(c / a, 0, 1); return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2); den = a * e - b * b
    if den > 1e-12:
        s = np.clip((b * f - c * e) / den, 0, 1)
        t = (b * s + f) / e
        if t < 0:
            t = 0; s = np.clip(c / a, 0, 1)
        elif t > 1:
            t = 1; s = np.clip((b + f) / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    t = np.clip(f / e, 0, 1); return np.linalg.norm(p1 - (p3 + t * d2))

def _verts(x, y, ang):
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    ang6 = np.array([0, 60, 120, 180, 240, 300]) * (math.pi / 180) + a
    return np.column_stack([ca * np.cos(ang6) - sa * np.sin(ang6),
                            sa * np.cos(ang6) + ca * np.sin(ang6)]) + np.array([x, y])

def _overlap_penalty(params, w=2000.0):
    n = N; pen = 0.0
    vs = [_verts(params[i, 0], params[i, 1], params[i, 2]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sep = _separation(vs[i], vs[j])
            if sep < 0.02:
                pen += w * (0.02 - sep) ** 2
    return pen

def _objective(params, outer_angle_deg):
    return _outer_side(params, outer_angle_deg) + _overlap_penalty(params)

def _valid(params, side, outer_angle_deg, min_sep=1e-6):
    # containment via support (exact)
    A = math.radians(outer_angle_deg)
    ca, sa = math.cos(A), math.sin(A)
    R = np.array([[ca, -sa], [sa, ca]])
    D = DIRS @ R.T
    for i in range(N):
        x, y, ang = params[i]
        a = math.radians(ang)
        ca2, sa2 = math.cos(a), math.sin(a)
        Rr = np.array([[ca2, -sa2], [sa2, ca2]])
        loc = DIRS @ Rr.T
        S = (loc * D).sum(axis=1) + (np.array([x, y]) @ D)
        if S.max() > side + 1e-7:
            return False
    # separation
    vs = [_verts(params[i, 0], params[i, 1], params[i, 2]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _separation(vs[i], vs[j]) < min_sep:
                return False
    return True

def _build(outer_r, inner_r, outer_angle_deg, inner_angle_deg, combo):
    p = np.zeros((N, 3))
    p[0] = [0.0, 0.0, inner_angle_deg]
    for k in range(6):
        ar = math.radians(k * 60)
        p[1 + k, 0] = inner_r * math.cos(ar)
        p[1 + k, 1] = inner_r * math.sin(ar)
        p[1 + k, 2] = inner_angle_deg
    for idx, ang_deg in enumerate(combo):
        ar = math.radians(ang_deg)
        p[7 + idx, 0] = outer_r * math.cos(ar)
        p[7 + idx, 1] = outer_r * math.sin(ar)
        p[7 + idx, 2] = outer_angle_deg
    return p

def _refine(params, outer_angle_deg, maxiter):
    try:
        res = minimize(_objective, params.flatten(), args=(outer_angle_deg,),
                       method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9})
        return res.x.reshape(N, 3), res.fun
    except Exception:
        return params, _objective(params, outer_angle_deg)

def _seed_solution():
    eps = 1e-9
    centers = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(k * 60)
        centers.append(((SQRT3 + eps) * math.cos(a), (SQRT3 + eps) * math.sin(a)))
    for k in range(4):
        a = math.radians(30 + k * 60)
        centers.append(((3.0 + 2 * eps) * math.cos(a), (3.0 + 2 * eps) * math.sin(a)))
    inner = [[x, y, 30.0] for x, y in centers]
    mx = (2 + eps) * SQRT3
    my = 4 + 2 * eps
    side = max(2 * mx / SQRT3, my) + 1e-9
    return inner, (0.0, 0.0), side, 30.0

def construct():
    n = N
    pointy = [30, 90, 150, 210, 270, 330]
    combos = list(combinations(range(6), 4))
    outer_angles = [30.0, 0.0, 15.0, 45.0]

    best_params = None
    best_side = float('inf')
    best_oa = 30.0

    for oa in outer_angles:
        if _elapsed() > DEADLINE - 20:
            break
        for combo in combos:
            if _elapsed() > DEADLINE - 20:
                break
            ang = [pointy[i] for i in combo]
            # quick 1-D bisection on outer ring radius to hit the non-overlap target
            # then a short local polish
            p = _build(3.0 + 2 * EPS, SQRT3 + EPS, oa, 30.0, ang)
            p, _ = _refine(p, oa, 120)
            s = _outer_side(p, oa)
            if _valid(p, s, oa, 1e-6):
                if s < best_side:
                    best_side = s
                    best_params = p.copy()
                    best_oa = oa

    if best_params is None:
        si, sc, ss, sa = _seed_solution()
        best_params = np.array(si)
        best_side = ss
        best_oa = sa

    # final polish on the best configuration
    if _elapsed() < DEADLINE - 10:
        p, _ = _refine(best_params, best_oa, 400)
        s = _outer_side(p, best_oa)
        if _valid(p, s, best_oa, 1e-6) and s < best_side:
            best_params = p
            best_side = s

    # final validity + fallback
    fs = _outer_side(best_params, best_oa)
    if not _valid(best_params, fs, best_oa, 1e-6):
        si, sc, ss, sa = _seed_solution()
        best_params = np.array(si)
        fs = ss
        best_oa = sa

    inner_rows = [[float(r[0]), float(r[1]), float(r[2])] for r in best_params]
    return inner_rows, (0.0, 0.0), float(fs), float(best_oa)
