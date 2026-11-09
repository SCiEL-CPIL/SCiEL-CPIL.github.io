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

def _build_objective(n):
    def objective(p):
        x = p[:2*n]
        angles = p[2*n:3*n]
        t = p[3*n]
        
        # Penalty for vertices outside outer hex
        pen = 0.0
        for i in range(n):
            xi, yi = x[2*i], x[2*i+1]
            ai = angles[i]
            for k in range(6):
                a = math.radians(60.0*k) + ai
                vx = xi + math.cos(a)
                vy = yi + math.sin(a)
                if not _point_in_hex(vx, vy, 0.0, 0.0, t, 0.0):
                    pen += 10.0
        
        # Penalty for overlap
        for i in range(n):
            vi = _hex_verts(x[2*i], x[2*i+1], 1.0, angles[i])
            for j in range(i+1, n):
                vj = _hex_verts(x[2*j], x[2*j+1], 1.0, angles[j])
                dc = math.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2)
                if dc < 4.0:
                    for m in range(6):
                        if _point_in_hex(vi[m,0], vi[m,1], x[2*j], x[2*j+1], 1.0, angles[j]):
                            pen += 10.0
                            break
                        if _point_in_hex(vj[m,0], vj[m,1], x[2*i], x[2*i+1], 1.0, angles[i]):
                            pen += 10.0
                            break
        return 0.5*t*t + pen
    return objective

def _honeycomb_positions():
    """Generate honeycomb packing positions for 11 hexagons."""
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # Ring 1: 6 hexagons at distance 2
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a)))
    # Ring 2: 4 hexagons at distance 4, alternating angles
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a)))
    return positions[:N]

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    objective = _build_objective(n)
    best_val = float('inf')
    best_p = None

    rng = np.random.RandomState(42)
    
    # Initial guess from honeycomb
    init_positions = _honeycomb_positions()
    
    # Try multiple trials with perturbation
    for trial in range(10):
        if _remaining() < 15.0:
            break

        p = np.zeros(3*n + 1)
        for i in range(n):
            p[2*i] = init_positions[i][0]
            p[2*i+1] = init_positions[i][1]
            p[2*n+i] = 0.0  # initial angles = 0
        p[3*n] = 5.0  # initial outer side length
        
        # Add perturbation for diversity
        if trial > 0:
            noise_scale = 0.3 * (trial / 10.0) + 0.1
            p[:3*n] += rng.randn(3*n) * noise_scale
            p[3*n] = 5.0 + rng.rand() * 1.0

        try:
            # Use Nelder-Mead for robustness
            res = minimize(objective, p, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    if best_p is None:
        # Fallback: linear arrangement
        inner = [[float(k*2.0), 0.0, 0.0] for k in range(n)]
        return inner, (0.0, 0.0), 40.0, 0.0

    x = best_p[:2*n]
    angles = best_p[2*n:3*n]
    t = best_p[3*n]

    inner = [[float(x[2*i]), float(x[2*i+1]), float(math.degrees(angles[i]))] for i in range(n)]

    # Validate and ensure reasonable output
    if t < 3.5 or t > 15.0:
        inner = [[float(k*2.0), 0.0, 0.0] for k in range(n)]
        t = 40.0

    return inner, (0.0, 0.0), float(t), 0.0

