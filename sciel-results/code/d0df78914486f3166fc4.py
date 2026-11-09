"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = 145.0
_C30 = math.cos(math.radians(30))

def _hex_inradius(s):
    return s * _C30

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _point_in_hex(px, py, cx, cy, s, rot):
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    d1 = abs(_C30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-_C30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _gjk_distance(A, B, max_iter=30):
    """
    Compute minimum distance between two convex polygons using GJK.
    Returns 0.0 if overlapping, else positive distance.
    """
    # Initial direction
    d = np.array([1.0, 0.0])
    simplex = []
    
    for _ in range(max_iter):
        # Support points
        a = A[np.argmax(A @ d)]
        b = B[np.argmax(B @ (-d))]
        p = a - b
        
        if np.dot(p, d) < 0:
            break
        
        simplex.append(p)
        
        if len(simplex) == 1:
            d = -simplex[0]
            if np.dot(d, d) < 1e-12:
                return 0.0
        elif len(simplex) == 2:
            p0, p1 = simplex
            v = p1 - p0
            w = -p0
            t = np.dot(w, v) / np.dot(v, v) if np.dot(v, v) > 1e-12 else 0.0
            if t >= 0 and t <= 1:
                closest = p0 + t * v
                if np.dot(closest, closest) < 1e-12:
                    return 0.0
                d = np.array([-v[1], v[0]])
                if np.dot(d, closest) > 0:
                    d = -d
            else:
                if t < 0:
                    d = -p0
                    simplex = [p0]
                else:
                    d = -p1
                    simplex = [p1]
        else:
            p0, p1, p2 = simplex
            v0 = p2 - p0
            v1 = p1 - p0
            w = -p0
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(w, v0)
            d21 = np.dot(w, v1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-12:
                d = -p0
                simplex = [p0]
                continue
            u = (d11 * d20 - d01 * d21) / denom
            v = (d00 * d21 - d01 * d20) / denom
            if u >= 0 and v >= 0 and u + v <= 1:
                return 0.0
            elif u >= 0 and v < 0:
                d = -p0
                simplex = [p0]
            elif u < 0 and v >= 0:
                d = -p0
                simplex = [p0]
            elif u >= 0 and v >= 0 and u + v > 1:
                d = -(p1 + p2) / 2
                simplex = [p1, p2]
            elif u < 0 and v < 0:
                d = -p0
                simplex = [p0]
            else:
                d = -p0
                simplex = [p0]
    
    if len(simplex) == 1:
        return float(np.linalg.norm(simplex[0]))
    elif len(simplex) == 2:
        p0, p1 = simplex
        v = p1 - p0
        t = np.dot(-p0, v) / np.dot(v, v) if np.dot(v, v) > 1e-12 else 0.0
        t = max(0, min(1, t))
        closest = p0 + t * v
        return float(np.linalg.norm(closest))
    else:
        return float(min(np.linalg.norm(p) for p in simplex))

def _is_valid_config(inner, center, side, angle):
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
            dist = _gjk_distance(vi, vj)
            if dist < 1e-7:
                return False
    return True

def _honeycomb_positions():
    positions = []
    positions.append((0.0, 0.0, 0.0))
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

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
    
    s_init = max_r / _C30
    s_init *= 1.01
    
    p0 = np.zeros(3*n + 1)
    for i in range(n):
        p0[3*i] = init_pos[i][0]
        p0[3*i+1] = init_pos[i][1]
        p0[3*i+2] = 0.0
    p0[3*n] = s_init
    
    best_val = float('inf')
    best_p = p0.copy()
    
    rng = np.random.RandomState(42)
    
    # Simulated annealing + local refinement
    # Phase 1: SA to explore configuration space
    T = 0.5
    T_min = 0.001
    cooling = 0.995
    
    current_p = p0.copy()
    current_val = float('inf')
    
    # Initial evaluation
    def _eval(p):
        x = p[0:3*n:3]
        y = p[1:3*n:3]
        angles = p[2:3*n:3]
        S = p[3*n]
        
        if S < 1.0:
            return 1e6
            
        pen = 0.0
        r_out = _hex_inradius(S)
        
        for i in range(n):
            verts = _hex_verts(x[i], y[i], 1.0, angles[i])
            for v in verts:
                dx = v[0]
                dy = v[1]
                d1 = abs(_C30 * dx + 0.5 * dy)
                d2 = abs(dy)
                d3 = abs(-_C30 * dx + 0.5 * dy)
                d_max = max(d1, d2, d3)
                if d_max > r_out:
                    pen += (d_max - r_out)**2
                    
        for i in range(n):
            vi = _hex_verts(x[i], y[i], 1.0, angles[i])
            for j in range(i+1, n):
                dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dc > 2.5:
                    continue
                vj = _hex_verts(x[j], y[j], 1.0, angles[j])
                d = _gjk_distance(vi, vj)
                if d < 0.1:
                    pen += 50.0 * (0.1 - d)**2
        
        return 0.5 * S * S + pen
    
    current_val = _eval(current_p)
    best_p = current_p.copy()
    best_val = current_val
    
    step_size = 0.3
    
    while _remaining() > 60.0:
        # Generate neighbor
        new_p = current_p.copy()
        new_p[0:3*n] += rng.randn(3*n) * step_size
        new_p[3*n] = max(1.0, new_p[3*n] + rng.randn() * 0.05)
        
        new_val = _eval(new_p)
        
        # Acceptance criterion
        if new_val < current_val or rng.rand() < math.exp(-(new_val - current_val) / T):
            current_p = new_p
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_p = current_p.copy()
        
        T *= cooling
        step_size *= 0.999
        
        if T < T_min:
            T = T_min
            step_size = 0.1
    
    # Phase 2: Local refinement with Nelder-Mead from best SA result
    if _remaining() > 20.0:
        try:
            res = minimize(_eval, best_p, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Multi-start local refinement from perturbed best
    if _remaining() > 15.0:
        for trial in range(3):
            if _remaining() < 10.0:
                break
            p_init = best_p.copy()
            p_init[0:3*n] += rng.randn(3*n) * 0.1
            try:
                res = minimize(_eval, p_init, method='Nelder-Mead',
                              options={'maxiter': 80, 'xatol': 1e-5, 'fatol': 1e-5})
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
        # Fallback: try to fix by slightly increasing S
        S_fixed = S * 1.05
        if _is_valid_config(inner, center, S_fixed, angle_out):
            return inner, center, float(S_fixed), angle_out
        else:
            inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
            return inner_init, center, float(s_init), angle_out
