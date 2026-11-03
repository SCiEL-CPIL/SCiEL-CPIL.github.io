"""Packing 12 unit regular hexagons into a minimal regular hexagon."""
import math
import time
import numpy as np

N = 12
DEADLINE = 135.0          # safety margin below the 150 s kill
TOL = 1e-6               # non-penetration tolerance for feasibility
EPS = 1e-9

# ---------- geometry helpers ----------
def _hex_verts(s):
    """Vertices of a regular hexagon (side s), flat-top orientation (0 deg)."""
    a = np.deg2rad(np.arange(0, 360, 60))
    return np.stack([s * np.cos(a), s * np.sin(a)], axis=1)

def _rot(pts, c):
    if abs(c) < 1e-12:
        return pts
    return pts @ c

def _hex_world(c, r, s):
    """World-space vertices of a unit hexagon: center c, rotation r (rad), side s."""
    v = _hex_verts(s)
    return _rot(v, r) + c

def _poly_sep(A, B, tol):
    """Positive separation distance between convex polygons A,B (SAT).
    Returns <=0 if they (nearly) touch/overlap, else the minimum gap."""
    n = len(A)
    m = len(B)
    best = np.inf
    for i in range(n):
        ax, ay = A[i]; bx, by = B[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L < 1e-12:
            continue
        nx, ny = -dy / L, dx / L
        proj = np.stack([A[:, 0] * nx + A[:, 1] * ny,
                         B[:, 0] * nx + B[:, 1] * ny], axis=1)
        amin, amax = proj[:, 0].min(), proj[:, 0].max()
        bmin, bmax = proj[:, 1].min(), proj[:, 1].max()
        if bmin > amin + tol:
            return bmin - amin
        if amin > bmin + tol:
            return amin - bmin
        gap = min(amin - bmin, bmin - amin)
        if gap < best:
            best = gap
    for i in range(m):
        ax, ay = A[i]; bx, by = B[(i + 1) % m]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L < 1e-12:
            continue
        nx, ny = -dy / L, dx / L
        proj = np.stack([A[:, 0] * nx + A[:, 1] * ny,
                         B[:, 0] * nx + B[:, 1] * ny], axis=1)
        amin, amax = proj[:, 0].min(), proj[:, 0].max()
        bmin, bmax = proj[:, 1].min(), proj[:, 1].max()
        if bmin > amin + tol:
            return bmin - amin
        if amin > bmin + tol:
            return amin - bmin
        gap = min(amin - bmin, bmin - amin)
        if gap < best:
            best = gap
    return best

def _inner_worlds(inner, s):
    """World-space vertex arrays for all inner hexagons (side s)."""
    out = []
    for (cx, cy, ang) in inner:
        out.append(_hex_world(np.array([cx, cy]), math.radians(ang), s))
    return out

def _outer_world(oc, R, oang):
    return _hex_world(np.array(oc), math.radians(oang), R)

def _feasible(inner, oc, R, oang, s, tol=TOL):
    """True if all inner hexagons (side s) are disjoint and inside outer hex."""
    oc = np.asarray(oc, dtype=float)
    inner_w = _inner_worlds(inner, s)
    outer_w = _outer_world(oc, R, oang)
    for A in inner_w:
        if _poly_sep(outer_w, A, tol) <= 0:
            return False
    for i in range(len(inner_w)):
        for j in range(i + 1, len(inner_w)):
            if _poly_sep(inner_w[i], inner_w[j], tol) <= 0:
                return False
    return True

# ---------- objective (minimisation) ----------
def _objective(x, oc, R, oang, s, tol=TOL):
    """Penalty: sum of overlaps (positive when infeasible). 0 => feasible."""
    inner = [(x[3 * i], x[3 * i + 1], x[3 * i + 2]) for i in range(N)]
    oc = np.asarray(oc, dtype=float)
    inner_w = _inner_worlds(inner, s)
    outer_w = _outer_world(oc, R, oang)
    pen = 0.0
    for A in inner_w:
        d = _poly_sep(outer_w, A, tol)
        if d < 0:
            pen += -d
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_sep(inner_w[i], inner_w[j], tol)
            if d < 0:
                pen += -d
    return pen

# ---------- initial layouts ----------
def _lattice(n):
    """Dense hexagonal lattice of N unit hexagons (flat-top, touching)."""
    dx = 3.0
    dy = math.sqrt(3.0)
    pts = []
    for row in range(60):
        for col in range(60):
            x = col * dx + (row % 2) * (dx / 2.0)
            y = row * dy
            pts.append((x, y))
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    return pts[:n]

def _make_inner(pts, ang=0.0):
    return [(px, py, ang) for (px, py) in pts]

# ---------- local refinement ----------
def _nelder_mead(f, x0, tol=1e-8, maxfev=800, deadline=None):
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float).copy()
        p[i] += 0.1 if x0[i] != 0 else 0.1
        simplex.append(p)
    vals = [f(p) for p in simplex]
    fev = n + 1
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    while fev < maxfev:
        if deadline is not None and time.time() > deadline:
            break
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol * (1 + abs(vals[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr); fev += 1
        if fr < vals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe); fev += 1
            simplex[-1] = xe if fe < fr else xr
            vals[-1] = min(fe, fr)
        elif fr < vals[-2]:
            simplex[-1] = xr; vals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc); fev += 1
            if fc < vals[-1]:
                simplex[-1] = xc; vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i]); fev += 1
    order = np.argsort(vals)
    return simplex[order[0]], vals[order[0]]

def _anneal(f, x0, t0=0.4, t1=0.01, iters=6000, deadline=None):
    cur = np.array(x0, dtype=float)
    curf = f(cur)
    best = cur.copy(); bestf = curf
    rng = np.random.default_rng(12345)
    for k in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        t = t0 * (t1 / t0) ** (k / max(1, iters - 1))
        cand = cur + rng.normal(0, 0.3 * (1.0 + t), size=len(cur))
        cf = f(cand)
        if cf < curf or rng.random() < math.exp(-(cf - curf) / max(t, 1e-12)):
            cur, curf = cand, cf
            if cf < bestf:
                best, bestf = cur.copy(), cf
    return best, bestf

# ---------- binary search on outer side ----------
def _bisect(inner, oc, oang, s, lo, hi, deadline):
    """Smallest outer side R (in [lo,hi]) admitting the fixed inner layout."""
    for _ in range(34):
        if time.time() > deadline:
            break
        mid = 0.5 * (lo + hi)
        if _feasible(inner, oc, mid, oang, s):
            hi = mid
        else:
            lo = mid
    return hi

def _refine(inner, oc, oang, s, lo, hi, deadline):
    """Minimise outer side over inner positions (and outer centre) for fixed layout."""
    def f(x):
        inner_x = [(x[3 * i], x[3 * i + 1], x[3 * i + 2]) for i in range(N)]
        ocx = np.array([x[3 * N], x[3 * N + 1]])
        return _bisect(inner_x, ocx, oang, s, lo, hi, deadline)
    x0 = np.array([c for (px, py, a) in inner for c in (px, py, a)] + list(oc))
    x, _ = _nelder_mead(f, x0, tol=1e-7, maxfev=250, deadline=deadline)
    x, _ = _anneal(f, x, t0=0.3, t1=0.02, iters=2500, deadline=deadline)
    x, _ = _nelder_mead(f, x, tol=1e-8, maxfev=400, deadline=deadline)
    inner_x = [(x[3 * i], x[3 * i + 1], x[3 * i + 2]) for i in range(N)]
    ocx = (x[3 * N], x[3 * N + 1])
    return inner_x, ocx, _bisect(inner_x, ocx, oang, s, lo, hi, deadline)

def construct():
    t0 = time.time()
    s = 1.0
    oang = 0.0
    oc = (0.0, 0.0)
    lo, hi = 3.0, 40.0

    starts = [
        ("lattice0", _make_inner(_lattice(N), 0.0)),
        ("lattice30", _make_inner(_lattice(N), 30.0)),
        ("lattice0b", _make_inner(_lattice(N), 0.0)),
    ]

    best = None
    for name, inner in starts:
        if time.time() > DEADLINE:
            break
        # ensure a feasible bracket
        if not _feasible(inner, oc, hi, oang, s):
            hi *= 1.5
        inner_x, ocx, R = _refine(inner, oc, oang, s, lo, hi, DEADLINE)
        if best is None or R < best[2]:
            best = (inner_x, ocx, R)

    if best is None:
        # fallback: trivially valid, large container
        inner = _make_inner(_lattice(N), 0.0)
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_x, ocx, R = best
    # final polish
    inner_x, ocx, R = _refine(inner_x, ocx, oang, s, lo, hi, DEADLINE)
    return inner_x, (float(ocx[0]), float(ocx[1])), float(R), float(oang)
