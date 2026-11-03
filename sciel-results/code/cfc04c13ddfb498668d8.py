import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, s, angle_rad):
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]) + angle_rad
    return np.column_stack([cx + s*np.cos(angles), cy + s*np.sin(angles)])

def sat_overlap(v1, v2):
    """Separating Axis Theorem for two convex polygons."""
    for poly in (v1, v2):
        for i in range(len(poly)):
            edge = poly[(i+1)%6] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            p1 = np.dot(v1, normal)
            p2 = np.dot(v2, normal)
            if np.max(p1) < np.min(p2) - 1e-9 or np.max(p2) < np.min(p1) - 1e-9:
                return False
    return True

def point_in_hexagon(pt, cx, cy, s, angle_rad):
    dx, dy = pt[0]-cx, pt[1]-cy
    ca, sa = np.cos(-angle_rad), np.sin(-angle_rad)
    lx, ly = ca*dx - sa*dy, sa*dx + ca*dy
    apothem = s * np.sqrt(3) / 2
    for k in range(6):
        a = (2*k+1) * np.pi / 6
        if lx*np.cos(a) + ly*np.sin(a) > apothem + 1e-9:
            return False
    return True

def objective(params):
    outer_cx, outer_cy, outer_s, outer_angle = params[0:4]
    if outer_s < 0.5:
        return 1e6
    
    penalty = 0.0
    inner_verts = []
    
    for i in range(N):
        off = 4 + i*3
        ix, iy, ia = params[off], params[off+1], params[off+2]
        verts = hex_vertices(ix, iy, 1.0, ia)
        inner_verts.append(verts)
        
        for v in verts:
            if not point_in_hexagon(v, outer_cx, outer_cy, outer_s, outer_angle):
                penalty += 5.0
                break
    
    for i in range(N):
        for j in range(i+1, N):
            if sat_overlap(inner_verts[i], inner_verts[j]):
                penalty += 5.0
    
    return outer_s + penalty

def construct():
    t0 = time.time()
    deadline = t0 + 130
    
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    pos = [i*v1 + j*v2 for i in range(3) for j in range(4)]
    cen = np.mean(pos, axis=0)
    pos = [p - cen for p in pos]
    
    params = [0.0, 0.0, 6.0, 0.0]
    for p in pos:
        params += [p[0], p[1], 0.0]
    
    best = params[:]
    bv = objective(params)
    
    try:
        r = minimize(objective, params, method='COBYLA', options={'maxiter':200, 'rhobeg':0.1})
        if time.time() < deadline and objective(r.x) < bv:
            best, bv = r.x, objective(r.x)
    except: pass
    
    if time.time() < deadline - 20:
        try:
            r2 = minimize(objective, best, method='COBYLA', options={'maxiter':300, 'rhobeg':0.01})
            if time.time() < deadline and objective(r2.x) < bv:
                best, bv = r2.x, objective(r2.x)
        except: pass
    
    ocx, ocy, os, oa = best[0:4]
    inner = [[best[4+i*3], best[5+i*3], np.degrees(best[6+i*3])] for i in range(N)]
    return inner, (ocx, ocy), os, np.degrees(oa)
