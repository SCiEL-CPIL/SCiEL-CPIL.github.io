"""Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon (flat-top at angle 0), side s, centered at (cx,cy)."""
    a = math.radians(angle_deg)
    v = np.empty((6, 2))
    for i in range(6):
        t = a + i * math.pi / 3.0
        v[i] = [cx + s * math.cos(t), cy + s * math.sin(t)]
    return v

def polygon_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygon_overlap_area(a, b):
    """Area of intersection of two convex polygons via Sutherland-Hodgman clip."""
    def clip(subject, clip_edge):
        # clip_edge: (p1, p2), keep points to the left of directed edge p1->p2
        p1, p2 = clip_edge
        out = []
        n = len(subject)
        for i in range(n):
            cur = subject[i]
            nxt = subject[(i + 1) % n]
            def inside(pt):
                return np.cross(p2 - p1, pt - p1) >= -1e-12
            cin = inside(cur)
            nin = inside(nxt)
            if cin:
                out.append(cur)
                if not nin:
                    # exit: intersection
                    d = p2 - p1
                    t = np.cross(p1 - cur, nxt - cur) / np.cross(d, nxt - cur)
                    out.append(cur + t * (nxt - cur))
            else:
                if nin:
                    # enter: intersection
                    d = p2 - p1
                    t = np.cross(p1 - cur, nxt - cur) / np.cross(d, nxt - cur)
                    out.append(cur + t * (nxt - cur))
        return np.array(out) if len(out) else np.empty((0, 2))

    # Clip a by each edge of b
    poly = a
    nb = len(b)
    for i in range(nb):
        p1 = b[i]
        p2 = b[(i + 1) % nb]
        poly = clip(poly, (p1, p2))
        if len(poly) < 3:
            return 0.0
    return polygon_area(poly)

def overlap_penalty(v, outer):
    """Sum of overlap areas of inner hexagons with outer (hard constraint)."""
    ox, oy, os, oa = outer
    outer_v = hex_vertices(ox, oy, os, oa)
    total = 0.0
    for i in range(N):
        ix, iy, ia = v[3*i:3*i+3]
        iv = hex_vertices(ix, iy, 1.0, ia)
        # cheap bounding-box reject
        if (iv[:,0].max() < outer_v[:,0].min() or iv[:,0].min() > outer_v[:,0].max() or
            iv[:,1].max() < outer_v[:,1].min() or iv[:,1].min() > outer_v[:,1].max()):
            continue
        total += polygon_overlap_area(iv, outer_v)
    return total

def objective(x):
    """x = [outer_x, outer_y, outer_s, outer_a_deg, (ix,iy,ia)*N]"""
    outer = (x[0], x[1], x[2], x[3])
    v = x[4:].reshape(N, 3)
    return x[2] + 10.0 * overlap_penalty(v, outer)

def inner_pair_overlap(v):
    """Sum of pairwise overlap areas among inner hexagons."""
    total = 0.0
    for i in range(N):
        vi = hex_vertices(v[i,0], v[i,1], 1.0, v[i,2])
        for j in range(i+1, N):
            vj = hex_vertices(v[j,0], v[j,1], 1.0, v[j,2])
            if (vi[:,0].max() < vj[:,0].min() or vi[:,0].min() > vj[:,0].max() or
                vi[:,1].max() < vj[:,1].min() or vi[:,1].min() > vj[:,1].max()):
                continue
            total += polygon_overlap_area(vi, vj)
    return total

def total_penalty(x):
    outer = (x[0], x[1], x[2], x[3])
    v = x[4:].reshape(N, 3)
    return overlap_penalty(v, outer) + inner_pair_overlap(v)

def make_seed(seed_type, s0):
    """Return initial x vector. seed_type in {'ring','linear','grid'}."""
    ox, oy, oa = 0.0, 0.0, 0.0
    inner = np.empty((N, 3))
    if seed_type == 'linear':
        # single row
        for i in range(N):
            inner[i] = [ (i - (N-1)/2.0) * 2.0, 0.0, 0.0 ]
    elif seed_type == 'grid':
        # 3 rows x 4 cols
        for i in range(N):
            r = i // 4
            c = i % 4
            inner[i] = [ (c - 1.5) * 2.0, (r - 1.0) * math.sqrt(3), 0.0 ]
    else:  # ring
        for i in range(N):
            t = 2*math.pi*i/N
            inner[i] = [ 2.5*math.cos(t), 2.5*math.sin(t), 0.0 ]
    # scale outer to roughly contain seed
    xs = np.abs(inner[:,0]); ys = np.abs(inner[:,1])
    s0 = max(2.0, np.max(xs)+1.5, np.max(ys)+1.5, s0)
    x = np.concatenate([[ox, oy, s0, oa], inner.ravel()])
    return x

def solve_one(x0, maxiter=400):
    """Run COBYLA on objective; return (x, obj)."""
    try:
        res = minimize(objective, x0, method='COBYLA',
                       options={'maxiter': maxiter, 'rhobeg': 1.0, 'catol': 1e-6})
        return res.x, res.fun
    except Exception:
        return x0, objective(x0)

def polish(x, maxiter=200):
    """Tighter local solve to reduce outer side."""
    try:
        res = minimize(objective, x, method='COBYLA',
                       options={'maxiter': maxiter, 'rhobeg': 0.3, 'catol': 1e-8})
        return res.x
    except Exception:
        return x

def construct():
    global DEADLINE
    best_x = None
    best_obj = float('inf')
    seeds = ['ring', 'linear', 'grid']
    for st in seeds:
        if time.time() > DEADLINE:
            break
        x0 = make_seed(st, 8.0)
        x, obj = solve_one(x0)
        # verify feasibility: penalty should be ~0
        if total_penalty(x) < 1e-6 and obj < best_obj:
            best_obj = obj
            best_x = x.copy()
        # polish
        if time.time() > DEADLINE:
            break
        xp = polish(x)
        if total_penalty(xp) < 1e-6:
            objp = objective(xp)
            if objp < best_obj:
                best_obj = objp
                best_x = xp.copy()

    if best_x is None:
        # fallback: return a valid (if not optimal) linear packing
        x0 = make_seed('linear', 12.0)
        best_x = x0

    ox, oy, os, oa = best_x[0], best_x[1], best_x[2], best_x[3]
    v = best_x[4:].reshape(N, 3)
    inner = [[float(v[i,0]), float(v[i,1]), float(v[i,2])] for i in range(N)]
    return inner, (float(ox), float(oy)), float(os), float(oa)
