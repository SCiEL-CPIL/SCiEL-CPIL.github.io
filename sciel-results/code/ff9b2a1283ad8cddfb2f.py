"""
Packing 12 unit regular hexagons inside a regular hexagon.
Strategy:
1. Analytic initial packing via hexagonal lattice.
2. Multi-start local search (Nelder-Mead) minimizing outer side length with penalties.
3. Return the best valid packing found within a time budget.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 110.0

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3.0
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def hexagon_vertices(cx, cy, side, angle_deg):
    return hex_vertices(cx, cy, side, angle_deg)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    angle_rad = -math.radians(angle_deg)
    lx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    ly = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        a = i * math.pi / 3.0
        nx = math.cos(a)
        ny = math.sin(a)
        if nx * lx + ny * ly > apothem + 1e-9:
            return False
    return True

def _segment_intersects(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def hexagons_disjoint(inner, tol=1e-6):
    n = len(inner)
    for i in range(n):
        xi, yi, ai = inner[i]
        vi = hex_vertices(xi, yi, 1.0, ai)
        for j in range(i + 1, n):
            xj, yj, aj = inner[j]
            vj = hex_vertices(xj, yj, 1.0, aj)
            for v in vi:
                if _point_in_hexagon(v[0], v[1], xj, yj, 1.0, aj):
                    return False
            for v in vj:
                if _point_in_hexagon(v[0], v[1], xi, yi, 1.0, ai):
                    return False
            for k in range(6):
                e1 = (vi[k], vi[(k+1)%6])
                for m in range(6):
                    e2 = (vj[m], vj[(m+1)%6])
                    if _segment_intersects(e1[0], e1[1], e2[0], e2[1]):
                        return False
    return True

def hexagon_in_hexagon(inner, center, side, angle, tol=1e-6):
    cx, cy = center
    for xi, yi, ai in inner:
        vi = hex_vertices(xi, yi, 1.0, ai)
        for v in vi:
            if not _point_in_hexagon(v[0], v[1], cx, cy, side, angle):
                return False
    return True

def is_valid(inner, center, side, angle):
    return hexagons_disjoint(inner) and hexagon_in_hexagon(inner, center, side, angle)

def _analytic_init():
    a1 = (math.sqrt(3), 0)
    a2 = (math.sqrt(3)/2, 3/2)
    lattice_points = [
        (0,0), (1,0), (2,0), (3,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2),
        (1,3)
    ]
    inner = []
    for (i, j) in lattice_points:
        x = i * a1[0] + j * a2[0]
        y = i * a1[1] + j * a2[1]
        inner.append([x, y, 0.0])
    cx = sum(p[0] for p in inner) / len(inner)
    cy = sum(p[1] for p in inner) / len(inner)
    max_r = 0
    for xi, yi, ai in inner:
        vi = hex_vertices(xi, yi, 1.0, ai)
        for v in vi:
            r = math.sqrt((v[0]-cx)**2 + (v[1]-cy)**2)
            max_r = max(max_r, r)
    side_est = max_r / (math.sqrt(3)/2) * 1.3
    return inner, (cx, cy), side_est, 0.0

def _objective(params):
    n = N
    inner = []
    for i in range(n):
        inner.append([params[3*i], params[3*i+1], params[3*i+2]])
    cx = params[3*n]
    cy = params[3*n+1]
    side = params[3*n+2]
    angle = params[3*n+3]
    
    penalty_contain = 0.0
    for xi, yi, ai in inner:
        vi = hex_vertices(xi, yi, 1.0, ai)
        for v in vi:
            if not _point_in_hexagon(v[0], v[1], cx, cy, side, angle):
                dx = v[0] - cx
                dy = v[1] - cy
                ar = -math.radians(angle)
                lx = dx * math.cos(ar) - dy * math.sin(ar)
                ly = dx * math.sin(ar) + dy * math.cos(ar)
                apothem = side * math.sqrt(3) / 2.0
                for k in range(6):
                    a = k * math.pi / 3.0
                    nx = math.cos(a)
                    ny = math.sin(a)
                    dist = nx * lx + ny * ly - apothem
                    if dist > 0:
                        penalty_contain += dist ** 2
    
    penalty_overlap = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            min_d = math.sqrt(3) - 0.05
            if d < min_d:
                penalty_overlap += (min_d - d) ** 2
    
    return side + 10.0 * penalty_contain + 10.0 * penalty_overlap

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    inner0, center0, side0, angle0 = _analytic_init()
    best_inner = [row[:] for row in inner0]
    best_center = center0[:]
    best_side = side0
    best_angle = angle0
    
    np.random.seed(42)
    n_starts = 30
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        inner_pert = []
        for x, y, a in inner0:
            inner_pert.append([x + np.random.randn()*0.3, 
                              y + np.random.randn()*0.3, 
                              a + np.random.randn()*3.0])
        
        x0 = []
        for x, y, a in inner_pert:
            x0.extend([x, y, a])
        x0.extend([center0[0], center0[1], side0, angle0])
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
            
            n = N
            inner_new = []
            for i in range(n):
                inner_new.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2]])
            cx_new = res.x[3*n]
            cy_new = res.x[3*n+1]
            side_new = res.x[3*n+2]
            angle_new = res.x[3*n+3]
            
            if is_valid(inner_new, (cx_new, cy_new), side_new, angle_new):
                if side_new < best_side:
                    best_inner = inner_new
                    best_center = (cx_new, cy_new)
                    best_side = side_new
                    best_angle = angle_new
        except Exception:
            pass
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
