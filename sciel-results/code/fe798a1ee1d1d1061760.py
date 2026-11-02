import numpy as np
from scipy.optimize import minimize
import time

N = 12
_T0 = time.time()
_DL = 145.0

def _hv(cx, cy, s, ang):
    a = np.radians(ang)
    return np.array([[cx + s*np.cos(a + i*np.pi/3), cy + s*np.sin(a + i*np.pi/3)] for i in range(6)])

def _mos(pos, oa):
    av = np.vstack([_hv(x, y, 1.0, a) for x, y, a in pos])
    best = 0.0
    o = np.radians(oa)
    for i in range(6):
        t = o + i*np.pi/3
        m = np.max(av[:,0]*np.cos(t) + av[:,1]*np.sin(t))
        best = max(best, m/(np.sqrt(3)/2))
    return best

def _grid(nc, nr):
    return [[c*1.5, r*np.sqrt(3)+(c%2)*np.sqrt(3)/2, 0.0] for r in range(nr) for c in range(nc)]

def _flower():
    p = [[0,0,0.0]]
    for i in range(6):
        a = i*np.pi/3
        p.append([np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a), 0.0])
    for i in range(5):
        a = i*2*np.pi/5 + np.pi/3
        p.append([2.5*np.sqrt(3)*np.cos(a), 2.5*np.sqrt(3)*np.sin(a), 0.0])
    return p

# EVOLVE-BLOCK-START
def construct():
    bp, bs, bo = None, float('inf'), 0.0
    inits = [_grid(4,3), _grid(3,4), _grid(6,2), _grid(2,6), _flower()]
    for cfg in inits:
        if time.time() > _DL: break
        cfg = cfg[:N]
        x0 = [v for p in cfg for v in p]
        def obj(p):
            pos = [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)]
            return _mos(pos, 0.0)
        try:
            r = minimize(obj, x0, method='Nelder-Mead', options={'maxiter':200,'xatol':1e-3,'fatol':1e-3})
            pos = [[r.x[3*i], r.x[3*i+1], r.x[3*i+2]] for i in range(N)]
            s = _mos(pos, 0.0)
            if s < bs:
                b2, o2 = s, 0.0
                for oa in range(0,180,3):
                    ss = _mos(pos, oa)
                    if ss < b2: b2, o2 = ss, oa
                for oa in np.linspace(o2-3, o2+3, 13):
                    ss = _mos(pos, oa)
                    if ss < b2: b2, o2 = ss, oa
                if b2 < bs: bs, bp, bo = b2, pos, o2
        except: pass
    if bp is None:
        return [[4.0*k,0.0,0.0] for k in range(-(N//2), N-N//2)], (0.0,0.0), 40.0, 0.0
    return bp, (0.0,0.0), bs, bo
# EVOLVE-BLOCK-END
