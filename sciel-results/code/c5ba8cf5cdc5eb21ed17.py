"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np

N = 11
_T0 = time.time()
_DEADLINE = 150.0

def _hexagon_vertices(cx, cy, s, rot):
    """Vertices of a regular hexagon (flat-top, rot=0) centered at (cx,cy)."""
    v = []
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        v.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return v

def _signed_dist(p, cx, cy, s, rot):
    """Positive if p is outside the hexagon, negative if inside (approx via support fn)."""
    dx = p[0] - cx
    dy = p[1] - cy
    # Transform into hexagon local frame
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # Hexagon (flat-top) half-width in x = s, and the 3 pairs of supporting lines:
    # |lx| <= s ; |sqrt3*ly| <= s*sqrt3 -> |ly| <= s ; |lx + ly| <= s ; |lx - ly| <= s
    # Actually for flat-top hexagon with circumradius s:
    # vertices at angles 0,60,...; edges perpendicular to angles 30,90,150,...
    # support functions: max of dot with unit normals at 30,90,150,210,270,330
    # Let's use the 3 axes: n1=(cos30,sin30), n2=(0,1), n3=(-cos30,sin30)
    n1x, n1y = math.cos(math.radians(30)), math.sin(math.radians(30))
    n2x, n2y = 0.0, 1.0
    n3x, n3y = -math.cos(math.radians(30)), math.sin(math.radians(30))
    d1 = n1x * lx + n1y * ly
    d2 = n2x * lx + n2y * ly
    d3 = n3x * lx + n3y * ly
    # Inradius of hexagon with circumradius s is s*cos(30)=s*sqrt3/2
    r_in = s * math.cos(math.radians(30))
    m = max(d1, -d1, d2, -d2, d3, -d3)
    return m - r_in

def _build_objective(n):
    """Return (objective, grad_fn) for packing n hexagons with outer side as t."""
    # params: x[2n], t[1]
    def objective(p):
        x = p[:2 * n]
        t = p[2 * n]
        cx = 0.0
        cy = 0.0
        rot = 0.0
        # outer side length = t
        # penalty for each inner hexagon vertex outside outer hexagon
        # and for pairwise overlap
        pen = 0.0
        # outer boundary: for each inner hex, all 6 vertices must be inside outer
        # use smooth penalty on signed distance
        for i in range(n):
            xi = x[2 * i]
            yi = x[2 * i + 1]
            for k in range(6):
                a = math.radians(60.0 * k)
                vx = xi + math.cos(a)
                vy = yi + math.sin(a)
                sd = _signed_dist((vx, vy), cx, cy, t, rot)
                if sd > 0:
                    pen += 100.0 * sd * sd
        # pairwise: each vertex of hex i inside hex j -> penalty
        for i in range(n):
            xi = x[2 * i]
            yi = x[2 * i + 1]
            for j in range(n):
                if i == j:
                    continue
                xj = x[2 * j]
                yj = x[2 * j + 1]
                # check vertices of i against j and vice versa (approx)
                for k in range(6):
                    a = math.radians(60.0 * k)
                    vx = xi + math.cos(a)
                    vy = yi + math.sin(a)
                    sd = _signed_dist((vx, vy), xj, yj, 1.0, 0.0)
                    if sd > 0:
                        pen += 100.0 * sd * sd
        return 0.5 * t * t + pen
    return objective

def construct():
    t_start = time.time()
    n = N
    # Initial layout: hexagonal cluster of 11 (center + ring of 6 + partial)
    # Use axial coords for a hex cluster
    # 11 hexagons: center, 6 in first ring, 4 more in second ring (or linear)
    # Start with a compact arrangement
    positions = []
    # center
    positions.append((0.0, 0.0))
    # ring 1: 6 hexagons
    for k in range(6):
        a = math.radians(60.0 * k)
        positions.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
    # ring 2: 4 more
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0 * math.cos(a), 4.0 * math.sin(a)))

    # Build initial parameter vector
    p = np.zeros(2 * n + 1)
    for i in range(n):
        p[2 * i] = positions[i][0]
        p[2 * i + 1] = positions[i][1]
    p[2 * n] = 10.0  # initial outer side

    objective = _build_objective(n)

    # Simple gradient-free optimization: Nelder-Mead via scipy
    from scipy.optimize import minimize
    best_p = p.copy()
    best_val = objective(p)

    def _elapsed():
        return time.time() - t_start

    # Coarse to fine
    for trial in range(3):
        if _elapsed() > _DEADLINE - 5.0:
            break
        try:
            res = minimize(objective, p, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
            p = res.x.copy()
        except Exception:
            pass

    # Extract best
    x = best_p[:2 * n]
    t = best_p[2 * n]
    inner = []
    for i in range(n):
        inner.append([float(x[2 * i]), float(x[2 * i + 1]), 0.0])
    # Safety: if t is too small (invalid), clamp to a known-good value
    if t < 3.0:
        t = 40.0
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(n // 2), n - n // 2)]
    return inner, (0.0, 0.0), float(t), 0.0
