"""
Optimized packing of 12 unit hexagons in a minimal outer hexagon.
Structural idea: fast validity + Nelder-Mead local search (inspiration 1).
Refinement machinery: precise bounding-hex computation and multi-angle outer polish (current).
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # safety margin before the 150s kill
APOTHEM = math.sqrt(3) / 2.0  # inradius of unit hexagon

# --- Geometry helpers ---

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy), rotated by angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project(verts, axis):
    ax, ay = axis
    vals = np.array([v[0]*ax + v[1]*ay for v in verts])
    return vals.min(), vals.max()

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-12:
            continue
        axes.append((-dy/L, dx/L))
    return axes

def _hex_overlap(v1, v2, axes1, axes2):
    """SAT test: True if overlapping."""
    for axis in axes1:
        minA, maxA = _project(v1, axis)
        minB, maxB = _project(v2, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    for axis in axes2:
        minA, maxA = _project(v1, axis)
        minB, maxB = _project(v2, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _is_valid(cx_arr, cy_arr, angle_arr):
    """Fast validity check: circle pre-filter then exact SAT."""
    n = len(cx_arr)
    min_d_sq = (2 * APOTHEM) ** 2 - 1e-8
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        if np.any(dx*dx + dy*dy < min_d_sq):
            return False
    verts = []
    axes = []
    for i in range(n):
        v = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        verts.append(v)
        axes.append(_axes_from_polygon(v))
    for i in range(n):
        for j in range(i+1, n):
            if _hex_overlap(verts[i], verts[j], axes[i], axes[j]):
                return False
    return True

def _bounding_side(allpts, theta):
    """Side length of smallest regular hexagon (orientation theta) containing allpts."""
    norms = np.array([[math.cos(theta + math.pi/6.0 + k*math.pi/3.0),
                       math.sin(theta + math.pi/6.0 + k*math.pi/3.0)] for k in range(6)])
    P = allpts @ norms.T
    return 2.0 * float(np.max(P - P.min(axis=0))) / math.sqrt(3.0)

def _compute_best_outer(cx_arr, cy_arr, angle_arr):
    """Find minimal enclosing regular hexagon by trying rotations and re-centering."""
    all_verts = []
    for i in range(N):
        all_verts.extend(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    allpts = np.array(all_verts)
    cx = float(allpts[:, 0].mean())
    cy = float(allpts[:, 1].mean())
    best_side = float('inf')
    best_ang = 0.0
    for ang in np.linspace(0, math.pi/3, 121):
        b = _bounding_side(allpts - np.array([cx, cy]), ang)
        if b < best_side - 1e-9:
            best_side = b
            best_ang = float(ang)
    # re-center precisely for best angle
    norms = np.array([[math.cos(best_ang + math.pi/6.0 + k*math.pi/3.0),
                       math.sin(best_ang + math.pi/6.0 + k*math.pi/3.0)] for k in range(6)]).T
    P = allpts @ norms
    mid = 0.5 * (P.max(axis=0) + P.min(axis=0))
    # shift so that projections are centered at mid
    shift = np.linalg.pinv(norms) @ (mid - P.mean(axis=0))
    newc = np.array([cx, cy]) + shift
    final_side = _bounding_side(allpts - newc, best_ang)
    return float(newc[0]), float(newc[1]), float(final_side), float(best_ang)

# --- Objective for optimizer ---

def _objective(params):
    cx = params[:N]
    cy = params[N:2*N]
    ang = params[2*N:]
    if not _is_valid(cx, cy, ang):
        return 1e6
    _, _, side, _ = _compute_best_outer(cx, cy, ang)
    return side

# --- Initial configurations ---

def _init_flower():
    pts = [[0.0, 0.0, 0.0]]
    r1 = 2 * math.sqrt(3)
    for i in range(6):
        a = math.radians(60 * i)
        pts.append([r1 * math.cos(a), r1 * math.sin(a), 0.0])
    r2 = r1 * 1.732
    for deg in [30, 90, 150, 210, 270, 330]:
        a = math.radians(deg)
        pts.append([r2 * math.cos(a), r2 * math.sin(a), 0.0])
    return pts[:12]

def _init_lattice():
    pts = []
    d = 2.0 * math.sqrt(3)
    a1 = np.array([d, 0.0])
    a2 = np.array([d/2, d * math.sqrt(3)/2])
    for i in range(-1, 3):
        for j in range(-1, 2):
            p = i * a1 + j * a2
            pts.append([float(p[0]), float(p[1]), 0.0])
    c = np.array(pts).mean(axis=0)
    return [[float(p[0]-c[0]), float(p[1]-c[1]), 0.0] for p in pts]

def _init_ring():
    pts = [[0.0, 0.0, 0.0]]
    r = 2 * math.sqrt(3)
    for i in range(11):
        a = math.radians(360 * i / 11 + 15)
        pts.append([r * math.cos(a), r * math.sin(a), 0.0])
    return pts

def _init_random():
    return [[random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 60)] for _ in range(N)]

# --- Main ---

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    best_side = float('inf')
    best_inner = None
    best_outer = (0.0, 0.0)
    best_theta = 0.0

    initializers = [_init_flower, _init_lattice, _init_ring, _init_random]

    for init_func in initializers:
        if time.time() - t0 > DEADLINE - 10:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx = np.array([p[0] for p in cfg])
        cy = np.array([p[1] for p in cfg])
        ang = np.array([p[2] for p in cfg])

        if not _is_valid(cx, cy, ang):
            # Try to relax slightly
            for _ in range(50):
                if time.time() - t0 > DEADLINE - 10:
                    break
                cx += np.random.normal(0, 0.05, N)
                cy += np.random.normal(0, 0.05, N)
            if not _is_valid(cx, cy, ang):
                continue

        _, _, side, theta = _compute_best_outer(cx, cy, ang)
        if side < best_side:
            best_side = side
            best_inner = [[float(cx[i]), float(cy[i]), float(ang[i])] for i in range(N)]
            best_theta = theta

        # Nelder-Mead refinement
        x0 = np.concatenate([cx, cy, ang])
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6,
                                    'adaptive': True})
            if res.success and res.fun < best_side:
                cx_r = res.x[:N]
                cy_r = res.x[N:2*N]
                ang_r = res.x[2*N:]
                if _is_valid(cx_r, cy_r, ang_r):
                    _, _, s_r, t_r = _compute_best_outer(cx_r, cy_r, ang_r)
                    if s_r < best_side:
                        best_side = s_r
                        best_inner = [[float(cx_r[i]), float(cy_r[i]), float(ang_r[i])] for i in range(N)]
                        best_theta = t_r
        except Exception:
            pass

    if best_inner is None:
        # Fallback: return seed
        inner = _init_lattice()
        cx = np.array([p[0] for p in inner])
        cy = np.array([p[1] for p in inner])
        ang = np.array([p[2] for p in inner])
        ox, oy, side, theta = _compute_best_outer(cx, cy, ang)
        return inner, (ox, oy), side, math.degrees(theta)

    # Final outer polish
    cx = np.array([p[0] for p in best_inner])
    cy = np.array([p[1] for p in best_inner])
    ang = np.array([p[2] for p in best_inner])
    ox, oy, side, theta = _compute_best_outer(cx, cy, ang)
    best_side = side
    best_outer = (ox, oy)
    best_theta = theta

    return best_inner, best_outer, best_side, math.degrees(best_theta)
