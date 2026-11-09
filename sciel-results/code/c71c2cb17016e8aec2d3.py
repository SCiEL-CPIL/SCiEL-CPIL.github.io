import numpy as np
from scipy.optimize import minimize
import time

N = 11
_SQRT3 = np.sqrt(3.0)
_TOL = 1e-9

def _hex_verts(cx, cy, angle_deg):
    a = np.deg2rad(angle_deg)
    th = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + a
    return np.column_stack([np.cos(th), np.sin(th)]) + np.array([cx, cy])

def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > _TOL else np.zeros(2)

def _point_in_convex(p, P):
    s = 0
    for i in range(6):
        e = P[(i + 1) % 6] - P[i]
        cross = e[0] * (p[1] - P[i][1]) - e[1] * (p[0] - P[i][0])
        if cross > _TOL:
            s |= 1
        elif cross < -_TOL:
            s |= 2
    return not (s == 3)

def _min_dist_convex(P, Q):
    """Exact minimum distance between two convex polygons (>=0)."""
    best = float('inf')
    for i in range(6):
        a1 = P[i]; a2 = P[(i + 1) % 6]
        d1 = a2 - a1
        for j in range(6):
            b1 = Q[j]; b2 = Q[(j + 1) % 6]
            d2 = b2 - b1
            r = a1 - b1
            a = np.dot(d1, d1); c = np.dot(d2, d2)
            b = np.dot(d1, d2); d = np.dot(d1, r); e = np.dot(d2, r)
            denom = a * c - b * b
            s = t = 0.0
            if denom <= _TOL:
                if a <= _TOL:
                    s = 0.0
                else:
                    s = np.clip(-d / a, 0.0, 1.0)
                t = np.clip((b * s + e) / c if c > _TOL else 0.0, 0.0, 1.0)
            else:
                s = np.clip((b * e - c * d) / denom, 0.0, 1.0)
                t = (b * s + e) / c
                if t < 0.0:
                    t = 0.0
                    s = np.clip(-d / a, 0.0, 1.0)
                elif t > 1.0:
                    t = 1.0
                    s = np.clip((b - d) / a, 0.0, 1.0)
            ca = a1 + s * d1
            cb = b1 + t * d2
            dd = ca - cb
            dist = np.dot(dd, dd)
            if dist < best:
                best = dist
    best = np.sqrt(best)
    for V in (P, Q):
        for p in V:
            if _point_in_convex(p, Q):
                return 0.0
    for V in (Q, P):
        for p in V:
            if _point_in_convex(p, P):
                return 0.0
    return best

def _containment_dist(P, outer):
    """Signed distance: positive if P inside outer hexagon, negative if outside."""
    mn = float('inf')
    for i in range(6):
        v1 = outer[i]; v2 = outer[(i + 1) % 6]
        n = _norm(np.array([-(v2[1] - v1[1]), v2[0] - v1[0]]))
        if np.dot(n, v1) > 0:
            n = -n
        for p in P:
            mn = min(mn, np.dot(p - v1, n))
    return mn

def _honeycomb_positions():
    """Compact hexagonal tiling of 11 flat-top unit hexagons, centered."""
    S = _SQRT3
    pos = []
    for r in range(3):
        for c in range(4):
            if len(pos) >= N:
                break
            x = c * 2 * S + (S if r % 2 else 0.0)
            y = r * 1.5
            pos.append([x, y])
    px = np.array([p[0] for p in pos]); py = np.array([p[1] for p in pos])
    cx = (px.max() + px.min()) / 2; cy = (py.max() + py.min()) / 2
    return np.array([[p[0] - cx, p[1] - cy] for p in pos])

def _penalty_objective(x, beta=50.0):
    """
    Smooth objective: outer_side + penalty for violations.
    x format: [x0, y0, a0, x1, y1, a1, ..., xN-1, yN-1, aN-1, side, angle]
    """
    pos = x[:3 * N].reshape(N, 2)
    ang = x[2::3][:N]
    outer_side = x[-2]
    outer_angle = x[-1]
    
    outer = _hex_verts(0.0, 0.0, outer_angle) * outer_side
    
    penalty = 0.0
    # Containment penalties
    for i in range(N):
        P = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
        d = _containment_dist(P, outer)
        if d < 0:
            penalty += d * d
    
    # Overlap penalties
    for i in range(N):
        Pi = _hex_verts(pos[i, 0], pos[i, 1], ang[i])
        for j in range(i + 1, N):
            Pj = _hex_verts(pos[j, 0], pos[j, 1], ang[j])
            d = _min_dist_convex(Pi, Pj)
            if d < 0.1:  # Only penalize if close
                penalty += (0.1 - d) * (0.1 - d)
    
    return outer_side + beta * penalty

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s limit
    rng = np.random.default_rng(42)
    base = _honeycomb_positions()
    
    def pack(x):
        pos = x[:3 * N].reshape(N, 2)
        ang = x[2::3][:N]
        outer_side = x[-2]
        outer_angle = x[-1]
        return pos, ang, outer_side, outer_angle

    def is_valid(pos, ang, outer_side, outer_angle):
        outer = _hex_verts(0.0, 0.0, outer_angle) * outer_side
        for i in range(N):
            if _containment_dist(_hex_verts(pos[i, 0], pos[i, 1], ang[i]), outer) < -1e-6:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                if _min_dist_convex(_hex_verts(pos[i, 0], pos[i, 1], ang[i]),
                                    _hex_verts(pos[j, 0], pos[j, 1], ang[j])) < -1e-6:
                    return False
        return True

    def refine(x, time_left):
        """Local refinement using Nelder-Mead on penalty objective."""
        if time_left < 2.0:
            return x
        try:
            r = minimize(_penalty_objective, x, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6,
                                  'adaptive': True, 'maxfev': 500})
            if is_valid(*pack(r.x)):
                return r.x
        except Exception:
            pass
        return x

    best = None
    best_side = float('inf')
    max_extent = max(np.max(np.abs(base)), 1.0) + 3.0
    
    # Initial guess
    x0 = np.zeros(3 * N + 2)
    for i in range(N):
        x0[3 * i] = base[i, 0]
        x0[3 * i + 1] = base[i, 1]
        x0[3 * i + 2] = 0.0
    x0[-2] = max_extent
    x0[-1] = 0.0

    # Multi-start with different perturbation scales and angles
    restarts = 25
    for restart in range(restarts):
        if time.time() > deadline - 30.0:
            break
        
        xstart = x0.copy()
        if restart > 0:
            scale = 1.0 + restart * 0.3
            for i in range(N):
                xstart[3 * i] += rng.uniform(-scale, scale)
                xstart[3 * i + 1] += rng.uniform(-scale, scale)
                xstart[3 * i + 2] += rng.uniform(-90, 90)
            xstart[-2] = max_extent + rng.uniform(-1.0, 3.0)
            xstart[-1] = rng.uniform(-30, 30)
        
        time_left = deadline - time.time() - 5.0
        if time_left < 3.0:
            break
            
        cand = refine(xstart, time_left)
        
        if is_valid(*pack(cand)):
            side = cand[-2]
            if side < best_side:
                best = cand.copy()
                best_side = side
                # Try to further refine the best found
                if time.time() < deadline - 10.0:
                    best = refine(best, deadline - time.time() - 5.0)
                    if is_valid(*pack(best)):
                        best_side = best[-2]

    if best is not None:
        pos, ang, outer_side, outer_angle = pack(best)
        inner = [[float(pos[i, 0]), float(pos[i, 1]), float(ang[i])] for i in range(N)]
        return inner, (0.0, 0.0), float(outer_side), float(outer_angle)
    else:
        # Fallback: return honeycomb with large enough container
        inner = [[float(base[i, 0]), float(base[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(max_extent + 5.0), 0.0
