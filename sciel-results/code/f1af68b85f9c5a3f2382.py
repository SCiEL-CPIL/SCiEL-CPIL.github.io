import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def hex_verts(cx, cy, side, angle_deg):
    """Vertices of regular hexagon."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def sat_sep(v1, v2):
    """SAT minimum separation between two convex polygons. Returns (sep, axis)."""
    axes = []
    for v in (v1, v2):
        for i in range(6):
            e = v[(i+1)%6] - v[i]
            el = np.linalg.norm(e)
            if el < 1e-12:
                continue
            axes.append(np.array([-e[1], e[0]]) / el)
    min_sep = np.inf
    best_axis = None
    for ax in axes:
        p1 = v1 @ ax
        p2 = v2 @ ax
        sep = max(p1.min() - p2.max(), p2.min() - p1.max())
        if sep < min_sep:
            min_sep = sep
            best_axis = ax
    return min_sep, best_axis

def outer_side(centers, oc, oa):
    """Minimum outer hex side to contain all inner hexagons."""
    oa_r = np.radians(oa)
    c, s = np.cos(oa_r), np.sin(oa_r)
    max_d = 0.0
    for (px, py, pa) in centers:
        v = hex_verts(px, py, 1.0, pa)
        rel = v - np.array([oc[0], oc[1]])
        # Rotate to outer frame
        rx = rel[:, 0] * c + rel[:, 1] * s
        ry = -rel[:, 0] * s + rel[:, 1] * c
        for k in range(6):
            th = np.radians(60 * k)
            nx, ny = np.cos(th), np.sin(th)
            d = np.abs(rx * nx + ry * ny)
            if d.max() > max_d:
                max_d = d.max()
    return max_d / (np.sqrt(3) / 2)

def all_seps(centers):
    """All pairwise SAT separations. Returns list of (i, j, sep, axis)."""
    n = len(centers)
    verts = [hex_verts(c[0], c[1], 1.0, c[2]) for c in centers]
    results = []
    for i in range(n):
        for j in range(i+1, n):
            sep, ax = sat_sep(verts[i], verts[j])
            results.append((i, j, sep, ax))
    return results

def energy(centers, oc, oa):
    """Compute (outer_side, overlap_penalty, forces)."""
    n = len(centers)
    forces = np.zeros((n, 3))
    
    # Overlap
    seps = all_seps(centers)
    overlap_pen = 0.0
    for (i, j, sep, ax) in seps:
        if sep < 0:
            overlap_pen += (-sep) ** 2
            push = -sep * 2.0
            if ax is not None:
                forces[i, 0] -= ax[0] * push
                forces[i, 1] -= ax[1] * push
                forces[j, 0] += ax[0] * push
                forces[j, 1] += ax[1] * push
    
    # Containment
    side = outer_side(centers, oc, oa)
    oa_r = np.radians(oa)
    c, s = np.cos(oa_r), np.sin(oa_r)
    limit = side * np.sqrt(3) / 2
    
    for i in range(n):
        px, py, pa = centers[i]
        v = hex_verts(px, py, 1.0, pa)
        rel = v - np.array([oc[0], oc[1]])
        rx = rel[:, 0] * c + rel[:, 1] * s
        ry = -rel[:, 0] * s + rel[:, 1] * c
        for k in range(6):
            th = np.radians(60 * k)
            nx, ny = np.cos(th), np.sin(th)
            d = rx * nx + ry * ny
            excess = np.maximum(d - limit, 0) + np.maximum(-d - limit, 0)
            if excess.sum() > 0:
                # Force in local frame
                fx_l = -nx * excess.sum()
                fy_l = -ny * excess.sum()
                # Rotate back
                forces[i, 0] += fx_l * c - fy_l * s
                forces[i, 1] += fx_l * s + fy_l * c
    
    return side, overlap_pen, forces

def is_valid(centers, oc, oa, tol=1e-8):
    """Check validity."""
    side = outer_side(centers, oc, oa)
    if side > outer_side(centers, oc, oa) + 1e-9:
        return False
    seps = all_seps(centers)
    for (i, j, sep, ax) in seps:
        if sep < -tol:
            return False
    return True

def init_hex():
    """Hexagonal lattice."""
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    pos = [(0,0), (dx,0), (-dx,0),
           (dx/2, dy*0.866), (-dx/2, dy*0.866),
           (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
           (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
           (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
           (0, dy*1.732)]
    return [[x, y, 0.0] for x, y in pos]

def init_ring(r=2.5):
    """Circular arrangement."""
    cs = []
    for i in range(N):
        th = 2*np.pi*i/N
        cs.append([r*np.cos(th), r*np.sin(th), np.degrees(th)])
    return cs

def init_random(seed):
    np.random.seed(seed)
    return [[np.random.uniform(-4,4), np.random.uniform(-4,4), np.random.uniform(0,60)] for _ in range(N)]

def sa_optimize(init, time_limit=20.0, T0=1.0, Tend=0.01):
    """Simulated annealing with fast energy."""
    t0 = time.time()
    centers = [list(c) for c in init]
    oc = [0.0, 0.0]
    oa = 0.0
    
    side, op, _ = energy(centers, oc, oa)
    cur = side + 5.0 * op
    best_c = [list(c) for c in centers]
    best_oc, best_oa = list(oc), oa
    best = cur
    
    T = T0
    n = len(centers)
    
    while time.time() - t0 < time_limit:
        new_c = [list(c) for c in centers]
        new_oc = list(oc)
        new_oa = oa
        
        i = np.random.randint(n)
        ss = T * 0.3
        new_c[i][0] += np.random.normal(0, ss)
        new_c[i][1] += np.random.normal(0, ss)
        new_c[i][2] = (new_c[i][2] + np.random.normal(0, ss*20)) % 60
        
        if np.random.random() < 0.05:
            new_oc[0] += np.random.normal(0, ss*0.2)
            new_oc[1] += np.random.normal(0, ss*0.2)
        if np.random.random() < 0.03:
            new_oa = (new_oa + np.random.normal(0, ss*5)) % 360
        
        ns, nop, _ = energy(new_c, new_oc, new_oa)
        nscore = ns + 5.0 * nop
        
        d = nscore - cur
        if d < 0 or np.random.random() < np.exp(-d / max(T, 1e-10)):
            centers, oc, oa = new_c, new_oc, new_oa
            cur = nscore
            if cur < best:
                best = cur
                best_c = [list(c) for c in centers]
                best_oc, best_oa = list(oc), oa
        
        T = max(Tend, T * 0.9999)
    
    # Recenter
    cx = np.mean([c[0] for c in best_c])
    cy = np.mean([c[1] for c in best_c])
    for c in best_c:
        c[0] -= cx
        c[1] -= cy
    best_oc = [best_oc[0] - cx, best_oc[1] - cy]
    
    return best_c, best_oc, outer_side(best_c, best_oc, best_oa), best_oa

def lbfghs_polish(centers, oc, oa, time_limit=25.0):
    """L-BFGS-B polish with analytical gradient for containment."""
    t0 = time.time()
    n = N
    p = np.zeros(3*n + 3)
    for i in range(n):
        p[3*i] = centers[i][0]
        p[3*i+1] = centers[i][1]
        p[3*i+2] = centers[i][2]
    p[3*n] = oc[0]
    p[3*n+1] = oc[1]
    p[3*n+2] = oa
    
    def obj(p):
        cs = [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(n)]
        o_c = [p[3*n], p[3*n+1]]
        o_a = p[3*n+2]
        side = outer_side(cs, o_c, o_a)
        seps = all_seps(cs)
        op = sum((-s)**2 for (_,_,s,_) in seps if s < 0)
        return side + 10.0 * op
    
    init_val = obj(p)
    
    def obj_grad(p):
        val = obj(p)
        g = np.zeros_like(p)
        h = 1e-5
        for i in range(len(p)):
            pp = p.copy(); pp[i] += h
            pm = p.copy(); pm[i] -= h
            g[i] = (obj(pp) - obj(pm)) / (2*h)
        return val, g
    
    result = minimize(obj_grad, p, method='L-BFGS-B', jac=True,
                      options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
    
    if time.time() - t0 > time_limit:
        return centers, oc, oa, oa
    
    np_ = result.x
    nc = [[np_[3*i], np_[3*i+1], np_[3*i+2]] for i in range(n)]
    noc = [np_[3*n], np_[3*n+1]]
    noa = np_[3*n+2]
    
    # Validate
    if not is_valid(nc, noc, noa):
        return centers, oc, oa, oa
    
    ns = outer_side(nc, noc, noa)
    os_ = outer_side(centers, oc, oa)
    if ns < os_:
        return nc, noc, ns, noa
    return centers, oc, oa, oa

def construct():
    t0 = time.time()
    best = None
    best_score = np.inf
    
    configs = [
        init_hex(),
        init_ring(2.5),
        init_ring(3.0),
        init_random(42),
        init_random(123),
        init_random(777),
    ]
    
    # Phase 1: SA on multiple starts
    for cfg in configs:
        if time.time() - t0 > DEADLINE - 35:
            break
        try:
            c, oc, side, oa = sa_optimize(cfg, time_limit=min(18, DEADLINE - (time.time()-t0) - 30))
            if is_valid(c, oc, oa):
                if side < best_score:
                    best_score = side
                    best = (c, tuple(oc), side, oa)
        except:
            pass
    
    # Phase 2: Polish
    if best is not None and time.time() - t0 < DEADLINE - 8:
        try:
            c, oc, oa = best[0], list(best[1]), best[3]
            pc, poc, ps, poa = lbfghs_polish(c, oc, oa, time_limit=min(20, DEADLINE - (time.time()-t0) - 3))
            if is_valid(pc, poc, poa):
                if ps < best_score:
                    best_score = ps
                    best = (pc, tuple(poc), ps, poa)
        except:
            pass
    
    if best is not None:
        return best
    
    # Fallback
    cs = init_ring(4.0)
    s = outer_side(cs, (0,0), 0)
    return cs, (0.0, 0.0), s, 0.0
