import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 110.0

def hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angs = a + np.pi/6 + np.arange(6)*np.pi/3
    return np.array([[cx + side*np.cos(t), cy + side*np.sin(t)] for t in angs])

def point_in_hex(px, py, cx, cy, side, angle_deg):
    v = hex_verts(cx, cy, side, angle_deg)
    for i in range(6):
        x1,y1 = v[i]; x2,y2 = v[(i+1)%6]
        ex,ey = x2-x1, y2-y1
        if ex*(py-y1) - ey*(px-x1) < -1e-9:
            return False
    return True

def disjoint(i1, i2):
    v1 = hex_verts(i1[0], i1[1], 1.0, i1[2])
    v2 = hex_verts(i2[0], i2[1], 1.0, i2[2])
    for p in v1:
        if point_in_hex(p[0], p[1], i2[0], i2[1], 1.0, i2[2]):
            return False
    for p in v2:
        if point_in_hex(p[0], p[1], i1[0], i1[1], 1.0, i1[2]):
            return False
    return True

def min_outer_side(inner, outer_angle):
    a = np.radians(outer_angle)
    normals = np.array([[np.cos(a+np.pi/6+k*np.pi/3), np.sin(a+np.pi/6+k*np.pi/3)] for k in range(6)])
    mx = 0.0
    for (ix,iy,ia) in inner:
        v = hex_verts(ix, iy, 1.0, ia)
        for p in v:
            for n in normals:
                val = p[0]*n[0] + p[1]*n[1]
                if val > mx: mx = val
    return 2.0*mx/np.sqrt(3.0)

def obj(params, oa):
    inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if not disjoint(inner[i], inner[j]):
                return 1e6
    return min_outer_side(inner, oa)

def construct():
    t0 = time.time()
    d = np.sqrt(3.0)
    # Hex cluster: center + ring1(6) + 4 from ring2
    pos = [(0,0)]
    for k in range(6):
        ang = np.radians(60*k)
        pos.append((d*np.cos(ang), d*np.sin(ang)))
    # 4 more: pick symmetric positions from ring2
    for k in [0, 1, 3, 4]:
        ang = np.radians(60*k)
        pos.append((2*d*np.cos(ang), 2*d*np.sin(ang)))
    # Center
    cx = np.mean([p[0] for p in pos]); cy = np.mean([p[1] for p in pos])
    pos = [(p[0]-cx, p[1]-cy) for p in pos]
    
    p0 = []
    for (x,y) in pos:
        p0.extend([x, y, 0.0])
    
    best_p = p0[:]
    best_s = 1e9
    best_a = 0.0
    
    for oa in range(0, 60, 5):
        s = obj(best_p, float(oa))
        if s < best_s:
            best_s = s; best_a = float(oa)
    
    # Local opt
    if time.time()-t0 < 70:
        try:
            r = minimize(obj, best_p, args=(best_a,), method='Nelder-Mead',
                        options={'maxiter':3000,'xatol':1e-9,'fatol':1e-9})
            if r.fun < best_s:
                best_p = r.x; best_s = r.fun
        except: pass
    
    for oa in range(0, 60, 2):
        s = obj(best_p, float(oa))
        if s < best_s:
            best_s = s; best_a = float(oa)
    
    # Polish
    if time.time()-t0 < 105:
        try:
            r = minimize(obj, best_p, args=(best_a,), method='Nelder-Mead',
                        options={'maxiter':2000,'xatol':1e-11,'fatol':1e-11})
            if r.fun < best_s:
                best_p = r.x; best_s = r.fun
        except: pass
    
    inner = [[float(best_p[3*i]), float(best_p[3*i+1]), float(best_p[3*i+2])] for i in range(N)]
    fs = min_outer_side(inner, best_a)
    
    if not np.isfinite(fs) or fs > 15:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, (0.0, 0.0), float(fs), float(best_a)

# EVOLVE-BLOCK-END
