import math
import time
import random
import numpy as np

N = 11
START = time.time()
DEADLINE = 148.0

def _hex_vertices(cx, cy, s, rot):
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    max_proj = 0.0
    for (x, y, a) in inner:
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / math.sqrt(3.0)

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_poly(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i+1) % n], pt)
        if abs(c) < 1e-9:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def _poly_overlap(A, B):
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            if maxA + 1e-6 < minB or maxB + 1e-6 < minA:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _hexagonal_seed():
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _make_seed(rng):
    inner = [[rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 60)] for _ in range(N)]
    return inner, (0.0, 0.0), 10.0, 0.0

def _try_refine(inner, outer_c, s, rot, rng, iters, budget):
    # Simulated Annealing to find a valid packing with minimal outer side
    # State: (inner_positions, outer_center, outer_rotation)
    # Objective: minimize outer_side, subject to no overlaps and containment
    
    def get_objective(state):
        inr, oc, rot = state
        s = _min_outer_side(inr, oc, rot)
        
        # Check validity and calculate penalty
        outer_pts = _hex_vertices(oc[0], oc[1], s, rot)
        polys = [_inner_hex(x, y, a) for (x, y, a) in inr]
        
        # Containment check
        for p in polys:
            if not _in_outer(p, outer_pts):
                # Penalty for being outside
                # Find max distance violation
                for pt in p:
                    # Check if pt is inside outer_pts
                    if not _point_in_poly(pt, outer_pts):
                        return s + 100.0 # High penalty
        
        # Overlap check
        for i in range(len(polys)):
            for j in range(i+1, len(polys)):
                if _poly_overlap(polys[i], polys[j]):
                    return s + 100.0 # High penalty
        
        return s

    cur_state = (inner, outer_c, rot)
    cur_obj = get_objective(cur_state)
    
    best_state = cur_state
    best_obj = cur_obj
    
    if cur_obj < 100.0: # If currently valid
         best_state = cur_state
         best_obj = cur_obj
    else:
         # Try to find a valid state
         best_state = None
         best_obj = float('inf')

    temp = 1.0
    cool_rate = 0.995
    
    for _ in range(iters):
        if time.time() > budget:
            break
        
        # Generate neighbor
        new_inr = []
        for (x, y, a) in cur_state[0]:
            new_inr.append([x + rng.gauss(0, 0.2), y + rng.gauss(0, 0.2), a + rng.gauss(0, 10)])
        
        new_oc = (cur_state[1][0] + rng.gauss(0, 0.1), cur_state[1][1] + rng.gauss(0, 0.1))
        new_rot = cur_state[2] + rng.gauss(0, 5)
        
        new_state = (new_inr, new_oc, new_rot)
        new_obj = get_objective(new_state)
        
        # Acceptance criteria
        delta = new_obj - cur_obj
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            cur_state = new_state
            cur_obj = new_obj
            
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_state = cur_state
                
        temp *= cool_rate
        
        if temp < 0.01:
            temp = 0.01

    # Final check: if best_state is invalid, return the initial valid seed if available or a safe default
    if best_state is None:
        return (inner, outer_c, s, rot)
    
    inr, oc, rot = best_state
    s_final = _min_outer_side(inr, oc, rot)
    
    # Strict final validation
    if _valid(inr, oc, s_final, rot):
        return (inr, oc, s_final, rot)
    else:
        # Fallback to original if it was valid, else try to tighten original
        if _valid(inner, outer_c, s, rot):
             s_tight = _min_outer_side(inner, outer_c, rot)
             if _valid(inner, outer_c, s_tight, rot):
                 return (inner, outer_c, s_tight, rot)
        # If all else fails, return a known valid configuration (hexagonal seed)
        # This is handled in construct() if needed, but let's return a safe tight version of input
        s_tight = _min_outer_side(inner, outer_c, rot)
        if _valid(inner, outer_c, s_tight, rot):
            return (inner, outer_c, s_tight, rot)
        return (inner, outer_c, s, rot)

def _nelder_mead(f, x0, bounds, max_iter=1000, tol=1e-8):
    n = len(x0)
    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        x[i] += 0.1
        simplex.append(x)
    
    best = min(simplex, key=f)
    
    for _ in range(max_iter):
        vals = [f(x) for x in simplex]
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        
        if abs(vals[-1] - vals[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
            else:
                simplex[-1] = xr
        elif fr < vals[-2]:
            simplex[-1] = xr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
        
        best = min(simplex, key=f)
    
    return best

def _objective(params, outer_c, rot, s_target):
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append([x, y, a])
    
    if not _valid(inner, outer_c, s_target, rot):
        return s_target + 10.0
    
    s_actual = _min_outer_side(inner, outer_c, rot)
    return s_actual

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 2.0
    rng = random.Random(42)
    
    seeds = []
    seeds.append(_hexagonal_seed())
    for _ in range(5):
        seeds.append(_make_seed(rng))
    
    for seed in seeds:
        if time.time() > budget:
            break
        inner, oc, s, rot = seed
        s_tight = _min_outer_side(inner, oc, rot)
        res = _try_refine(inner, oc, s_tight, rot, rng, iters=200, budget=budget)
        inner, oc, s, rot = res
        if _valid(inner, oc, s, rot):
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)
    
    # Nelder-Mead refinement on best solution
    if best is not None and time.time() < budget - 10.0:
        inner, oc, s, rot = best
        params = np.array([v for hex_ in inner for v in hex_])
        
        def obj(p):
            if time.time() > budget:
                return 100.0
            inr = [[p[i*3], p[i*3+1], p[i*3+2]] for i in range(N)]
            if not _valid(inr, oc, s, rot):
                return s + 5.0
            return _min_outer_side(inr, oc, rot)
        
        try:
            res_params = _nelder_mead(obj, params, None, max_iter=500, tol=1e-6)
            inr = [[res_params[i*3], res_params[i*3+1], res_params[i*3+2]] for i in range(N)]
            s_new = _min_outer_side(inr, oc, rot)
            if _valid(inr, oc, s_new, rot) and s_new < best_score:
                best_score = s_new
                best = (inr, oc, s_new, rot)
        except Exception:
            pass
    
    # More restarts
    restarts = 0
    while time.time() < budget - 5.0 and (best is None or best_score > 4.5):
        restarts += 1
        inner, oc, s, rot = _make_seed(rng)
        s_tight = _min_outer_side(inner, oc, rot)
        res = _try_refine(inner, oc, s_tight, rot, rng, iters=150, budget=budget)
        inner, oc, s, rot = res
        if _valid(inner, oc, s, rot):
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)
        if restarts >= 30:
            break
    
    if best is None:
        inner, oc, s, rot = _hexagonal_seed()
        s_tight = _min_outer_side(inner, oc, rot)
        if _valid(inner, oc, s_tight, rot):
            return inner, (oc[0], oc[1]), s_tight, rot
        else:
            # Should not happen for hexagonal seed, but just in case
            return inner, (oc[0], oc[1]), s, rot
    
    inner, oc, s, rot = best
    s_tight = _min_outer_side(inner, oc, rot)
    if _valid(inner, oc, s_tight, rot):
        s = s_tight
    elif _valid(inner, oc, s, rot):
        pass # Keep s
    else:
        # Invalid result, fallback to hexagonal seed
        inner, oc, s, rot = _hexagonal_seed()
        s_tight = _min_outer_side(inner, oc, rot)
        if _valid(inner, oc, s_tight, rot):
            s = s_tight
    
    return inner, (oc[0], oc[1]), s, rot
