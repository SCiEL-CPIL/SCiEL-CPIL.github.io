import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def _hex_verts(cx, cy, ang_deg):
    th = np.radians(ang_deg)
    a = th + np.radians(60 * np.arange(6))
    return np.column_stack([cx + np.cos(a), cy + np.sin(a)])

def _normals(ang_deg):
    th = np.radians(ang_deg)
    a = th + np.radians(30 + 60 * np.arange(6))
    return np.column_stack([np.cos(a), np.sin(a)])

def _edge_intersect(p1, p2, p3, p4):
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

def _pt_in_hex(p, hv):
    for i in range(6):
        a = hv[i]
        b = hv[(i+1) % 6]
        if (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) < -1e-9:
            return False
    return True

def _disjoint_check(pos, angs):
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d >= 2.0:
                continue
            vi = _hex_verts(pos[i,0], pos[i,1], angs[i])
            vj = _hex_verts(pos[j,0], pos[j,1], angs[j])
            for v in vi:
                if _pt_in_hex(v, vj):
                    return False
            for v in vj:
                if _pt_in_hex(v, vi):
                    return False
            for k in range(6):
                for m in range(6):
                    if _edge_intersect(vi[k], vi[(k+1)%6], vj[m], vj[(m+1)%6]):
                        return False
    return True

def _enclosing_side(pos, angs, center):
    all_v = np.vstack([_hex_verts(pos[i,0], pos[i,1], angs[i]) for i in range(N)])
    best = float('inf')
    for oa in np.linspace(0, 60, 13):
        nm = _normals(oa)
        proj = (all_v - center) @ nm.T
        mp = np.max(proj)
        S = mp / (SQRT3/2)
        if S < best:
            best = S
    return best

def _obj(x):
    pos = x[:N*2].reshape(N, 2)
    angs = x[N*2:]
    if not _disjoint_check(pos, angs):
        return 1e5
    c = np.mean(pos, axis=0)
    return _enclosing_side(pos, angs, c)

def construct():
    t0 = time.time()
    DEADLINE = t0 + 100
    
    pos0 = []
    ang0 = []
    dx = SQRT3
    dy = 1.5
    for r in range(3):
        for c in range(4):
            pos0.append([c*dx + (r%2)*dx/2, r*dy])
            ang0.append(0.0)
    x0 = np.concatenate([np.array(pos0).flatten(), np.array(ang0)])
    
    best_x = x0.copy()
    best_S = _obj(best_x)
    
    bounds = [(-8, 8)]*(N*2) + [(-30, 30)]*N
    
    if time.time() < DEADLINE - 20:
        try:
            r = differential_evolution(_obj, bounds, maxiter=30, popsize=12,
                                       seed=42, polish=False, tol=1e-3,
                                       mutation=(0.5,1.0), recombination=0.8)
            if r.fun < best_S:
                best_S = r.fun
                best_x = r.x
        except:
            pass
    
    if time.time() < DEADLINE - 10:
        try:
            r = minimize(_obj, best_x, method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol':1e-5, 'fatol':1e-5})
            if r.fun < best_S:
                best_S = r.fun
                best_x = r.x
        except:
            pass
    
    pos = best_x[:N*2].reshape(N, 2)
    angs = best_x[N*2:]
    c = np.mean(pos, axis=0)
    S = _enclosing_side(pos, angs, c)
    
    inner = [[float(pos[i,0]), float(pos[i,1]), float(angs[i])] for i in range(N)]
    return inner, (float(c[0]), float(c[1])), float(S), 0.0
