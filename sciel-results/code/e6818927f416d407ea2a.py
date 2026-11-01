"""Packing 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, s, angle_deg):
    a = np.deg2rad(angle_deg)
    angs = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def point_in_hex(pt, cx, cy, s, angle_deg):
    v = hex_vertices(cx, cy, s, angle_deg)
    x, y = pt
    sign = 0
    for i in range(6):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % 6]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross > 1e-9:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-9:
            if sign > 0:
                return False
            sign = -1
    return True

def hexagon_inside_hexagon(oc, os_, oa, ic, ia):
    iv = hex_vertices(ic[0], ic[1], 1.0, ia)
    for v in iv:
        if not point_in_hex(v, oc[0], oc[1], os_, oa):
            return False
    return True

def hex_disjoint(c1, a1, c2, a2):
    d = np.linalg.norm(np.array(c1) - np.array(c2))
    if d < np.sqrt(3) - 1e-7:
        return False
    if d > 2.01:
        return True
    v1 = hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = hex_vertices(c2[0], c2[1], 1.0, a2)
    for v in v1:
        if point_in_hex(v, c2[0], c2[1], 1.0, a2):
            return False
    for v in v2:
        if point_in_hex(v, c1[0], c1[1], 1.0, a1):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    sq3 = np.sqrt(3.0)
    
    # Hex grid layout: 4 cols x 3 rows, flat-top
    positions = []
    for col in range(4):
        for row in range(3):
            x = col * 1.5
            y = row * sq3
            if col % 2 == 1:
                y += sq3 / 2.0
            positions.append([x, y, 0.0])
    
    # Center
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (min(xs) + max(xs)) / 2.0
    cy0 = (min(ys) + max(ys)) / 2.0
    for p in positions:
        p[0] -= cx0
        p[1] -= cy0
    
    def check_and_score(x, oa):
        """x = [x1,y1,a1,...,x12,y12,a12, ox,oy,os]"""
        oc = x[-3:-1]
        os_ = x[-1]
        if os_ < 1.0:
            return 1e6
        centers = [(x[3*i], x[3*i+1]) for i in range(N)]
        angles = [x[3*i+2] for i in range(N)]
        for i in range(N):
            if not hexagon_inside_hexagon(oc, os_, oa, centers[i], angles[i]):
                return 1e6
        for i in range(N):
            for j in range(i+1, N):
                if not hex_disjoint(centers[i], angles[i], centers[j], angles[j]):
                    return 1e6
        return os_
    
    best_s = 100.0
    best_x = None
    
    x0 = []
    for p in positions:
        x0.extend([p[0], p[1], p[2]])
    x0.extend([0.0, 0.0, 5.0])
    
    for oa in [0.0, 30.0]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(check_and_score, x0, args=(oa,), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-3, 'fatol': 1e-3})
            if res.fun < best_s and res.fun < 1e5:
                best_s = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: use initial layout with a generous outer hex
        return [[p[0], p[1], p[2]] for p in positions], (0.0, 0.0), 10.0, 0.0
    
    oc = best_x[-3:-1]
    os_ = best_x[-1]
    inner = [[best_x[3*i], best_x[3*i+1], best_x[3*i+2]] for i in range(N)]
    return inner, (float(oc[0]), float(oc[1])), float(os_), 0.0
# EVOLVE-BLOCK-END
