"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = 145.0

def _hex_inradius(s):
    return s * math.cos(math.radians(30))

def _point_in_hex(px, py, cx, cy, s, rot):
    """Check if point (px, py) is inside a hexagon centered at (cx, cy) with side s and rotation rot."""
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _min_dist_poly(A, B):
    """Compute minimum distance between two convex polygons (lists of 2D points)."""
    nA = len(A)
    nB = len(B)
    
    def point_in_poly(pt, poly):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % len(poly)]
            nx = -(y2 - y1)
            ny = (x2 - x1)
            d = nx * (pt[0] - x1) + ny * (pt[1] - y1)
            if d < -1e-9:
                return False
        return True
    
    for v in A:
        if point_in_poly(v, B):
            return 0.0
    for v in B:
        if point_in_poly(v, A):
            return 0.0
    
    min_d = float('inf')
    for i in range(nA):
        ax1, ay1 = A[i]
        ax2, ay2 = A[(i+1) % nA]
        for j in range(nB):
            bx1, by1 = B[j]
            bx2, by2 = B[(j+1) % nB]
            
            def pt_seg_dist(px, py, x1, y1, x2, y2):
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    return math.sqrt((px-x1)**2 + (py-y1)**2)
                t = ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                qx = x1 + t*dx
                qy = y1 + t*dy
                return math.sqrt((px-qx)**2 + (py-qy)**2)
            
            d1 = pt_seg_dist(ax1, ay1, bx1, by1, bx2, by2)
            d2 = pt_seg_dist(ax2, ay2, bx1, by1, bx2, by2)
            d3 = pt_seg_dist(bx1, by1, ax1, ay1, ax2, ay2)
            d4 = pt_seg_dist(bx2, by2, ax1, ay1, ax2, ay2)
            d = min(d1, d2, d3, d4)
            if d < min_d:
                min_d = d
    
    return min_d

def _is_valid_config(inner, center, side, angle):
    """Strictly validate the configuration."""
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False

    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.0:
                continue
            
            overlap = False
            for v in vi:
                if _point_in_hex(v[0], v[1], xj, yj, 1.0, aj):
                    overlap = True
                    break
            if overlap:
                return False
            
            for v in vj:
                if _point_in_hex(v[0], v[1], xi, yi, 1.0, ai):
                    overlap = True
                    break
            if overlap:
                return False
                
    return True

def _honeycomb_positions():
    """Generate honeycomb packing positions for 11 hexagons."""
    positions = []
    positions.append((0.0, 0.0, 0.0))
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

def _logsumexp(vals, beta=50.0):
    """Compute log-sum-exp for smooth approximation of max."""
    if not vals:
        return 0.0
    max_v = max(vals)
    if max_v == float('-inf'):
        return 0.0
    exp_sum = sum(math.exp(beta * (v - max_v)) for v in vals)
    return max_v + math.log(exp_sum) / beta

def _smooth_penalty_for_pair(vi, vj):
    """Compute smooth penalty for overlap between two hexagons."""
    dist = _min_dist_poly(vi, vj)
    if dist < 1e-6:
        return 5.0
    elif dist < 0.1:
        return (0.1 - dist) * 2.0
    else:
        return 0.0

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    init_pos = _honeycomb_positions()
    
    max_r = 0.0
    for (xi, yi, ai) in init_pos:
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r > max_r:
                max_r = r
    
    s_init = max_r / math.cos(math.radians(30))
    s_init *= 1.01
    
    p0 = np.zeros(3*n + 1)
    for i in range(n):
        p0[3*i] = init_pos[i][0]
        p0[3*i+1] = init_pos[i][1]
        p0[3*i+2] = 0.0
    p0[3*n] = s_init
    
    def objective(p):
        x = p[0:3*n:3]
        y = p[1:3*n:3]
        angles = p[2:3*n:3]
        S = p[3*n]
        
        if S < 1.0:
            return 1e6
            
        pen = 0.0
        r_out = _hex_inradius(S)
        c30 = math.cos(math.radians(30))
        
        # Boundary penalty
        for i in range(n):
            verts = _hex_verts(x[i], y[i], 1.0, angles[i])
            for v in verts:
                dx = v[0]
                dy = v[1]
                d1 = abs(c30 * dx + 0.5 * dy)
                d2 = abs(dy)
                d3 = abs(-c30 * dx + 0.5 * dy)
                d_max = max(d1, d2, d3)
                violation = max(0.0, d_max - r_out)
                pen += violation * violation
                    
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dc > 2.0:
                    continue
                vi = _hex_verts(x[i], y[i], 1.0, angles[i]).tolist()
                vj = _hex_verts(x[j], y[j], 1.0, angles[j]).tolist()
                
                pen += _smooth_penalty_for_pair(vi, vj)
        
        return 0.5 * S * S + 10.0 * pen

    best_val = float('inf')
    best_p = p0.copy()
    
    rng = np.random.RandomState(42)
    
    # First: Nelder-Mead from initial
    if _remaining() > 30.0:
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Second: Perturbed restarts
    for trial in range(4):
        if _remaining() < 25.0:
            break
            
        p_init = best_p.copy()
        noise = 0.15
        p_init[0:3*n] += rng.randn(3*n) * noise
        p_init[3*n] = max(1.0, s_init * (0.95 + rng.rand()*0.1))
        
        try:
            res = minimize(objective, p_init, method='Nelder-Mead',
                          options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Third: Polish pass with tighter tolerance
    if _remaining() > 15.0:
        try:
            res = minimize(objective, best_p, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    x = best_p[0:3*n:3]
    y = best_p[1:3*n:3]
    angles = best_p[2:3*n:3]
    S = best_p[3*n]
    
    inner = [[float(x[i]), float(y[i]), float(math.degrees(angles[i]))] for i in range(n)]
    center = (0.0, 0.0)
    angle_out = 0.0
    
    if _is_valid_config(inner, center, S, angle_out):
        return inner, center, float(S), angle_out
    else:
        for margin in [0.01, 0.02, 0.05, 0.1, 0.2]:
            S_try = S + margin
            if _is_valid_config(inner, center, S_try, angle_out):
                return inner, center, float(S_try), angle_out
        
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, center, float(s_init), angle_out
