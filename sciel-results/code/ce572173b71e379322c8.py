import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    return np.abs(det)

def clamp_bary(bary, eps=1e-10):
    u = np.clip(bary[:, 0].copy(), eps, 1.0)
    v = np.clip(bary[:, 1].copy(), eps, 1.0)
    s = u + v
    bad = s > 1.0 - eps
    if np.any(bad):
        scale = (1.0 - eps) / np.maximum(s[bad], eps)
        u[bad] *= scale
        v[bad] *= scale
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    return np.column_stack([u, v])

def min_area(bary):
    return np.min(calc_areas_bary(bary))

def hard_min_obj(flat):
    bary = clamp_bary(flat.reshape(N, 2))
    return -min_area(bary)

def soft_min_obj(flat, beta):
    bary = clamp_bary(flat.reshape(N, 2))
    areas = calc_areas_bary(bary)
    if np.any(areas < 1e-14):
        return 1e10
    s = -beta * areas
    mx = np.max(s)
    if mx > 700:
        return -np.min(areas)
    lse = mx + np.log(np.sum(np.exp(s - mx)))
    return -lse / beta

# 3-fold rotation in barycentric: (u,v) -> (w,u) -> (v,w) where w=1-u-v
def rot120(pt):
    u, v = pt
    w = 1.0 - u - v
    return np.array([w, u])

def rot240(pt):
    u, v = pt
    w = 1.0 - u - v
    return np.array([v, w])

def make_orbit(ref):
    ref = clamp_bary(ref.reshape(1, 2))[0]
    return np.array([ref, rot120(ref), rot240(ref)])

# Parameterized template: 1 centroid + 3 orbits (9 pts) + 1 axis point = 11
# Free params: r1, r2, r3 (orbit radii along one direction), t (axis point offset)
def template_sym(params):
    r1, r2, r3, t = params
    centroid = np.array([1/3, 1/3])
    # Orbits: reference point at centroid + r*(1, -0.5) direction in barycentric
    ref1 = np.array([1/3 + r1, 1/3 - r1/2])
    ref2 = np.array([1/3 + r2, 1/3 - r2/2])
    ref3 = np.array([1/3 + r3, 1/3 - r3/2])
    orb1 = make_orbit(ref1)
    orb2 = make_orbit(ref2)
    orb3 = make_orbit(ref3)
    # 11th point on the vertical symmetry axis (u = 1/3)
    axis_pt = np.array([1/3, 1/3 + t])
    pts = np.vstack([centroid.reshape(1,2), orb1, orb2, orb3, axis_pt.reshape(1,2)])
    return clamp_bary(pts)

def template_sym_obj(params):
    bary = template_sym(params)
    return -min_area(bary)

def gen_sym_template():
    """Optimize the symmetric template."""
    # Initial guess based on known good configurations
    best_bary = None
    best_val = -np.inf
    rng = np.random.default_rng(7)
    
    # Try multiple initial radii
    base_radii = [0.12, 0.22, 0.32]
    for trial in range(8):
        r1 = base_radii[0] + rng.uniform(-0.05, 0.05)
        r2 = base_radii[1] + rng.uniform(-0.05, 0.05)
        r3 = base_radii[2] + rng.uniform(-0.05, 0.05)
        t = rng.uniform(-0.15, 0.15)
        params0 = np.array([r1, r2, r3, t])
        
        # First optimize with soft min
        try:
            res = minimize(soft_min_obj_wrapper, params0, args=(30.0,),
                          method='L-BFGS-B',
                          bounds=[(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)],
                          options={'maxiter': 100, 'ftol': 1e-12})
            params0 = res.x
        except Exception:
            pass
        
        # Then hard min NM
        try:
            res2 = minimize(template_sym_obj, params0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
            params0 = res2.x
        except Exception:
            pass
        
        bary = template_sym(params0)
        val = min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    return best_bary

def soft_min_obj_wrapper(params, beta):
    bary = template_sym(params)
    areas = calc_areas_bary(bary)
    if np.any(areas < 1e-14):
        return 1e10
    s = -beta * areas
    mx = np.max(s)
    if mx > 700:
        return -np.min(areas)
    lse = mx + np.log(np.sum(np.exp(s - mx)))
    return -lse / beta

def gen_hex_init():
    """Hexagonal lattice subset."""
    best_bary = None
    best_val = -np.inf
    rng = np.random.default_rng(42)
    
    for scale in [0.18, 0.22, 0.26, 0.30]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts = int((1 - w_val) / scale) + 1
            for j in range(num_pts):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        if len(pts) >= N:
            for _ in range(3):
                idx = rng.choice(len(pts), N, replace=False)
                bary = np.array(pts)[idx]
                val = min_area(bary)
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    if best_bary is None:
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    return best_bary

def gen_random(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def gen_perturb(base, sigma, seed):
    rng = np.random.default_rng(seed)
    return clamp_bary(base + rng.normal(0, sigma, base.shape))

def refine_local(bary_init, time_limit=5.0):
    start = time.time()
    bounds = [(0, 1)] * (N * 2)
    cur = bary_init.copy()
    
    # Stage 1: soft min beta=10
    try:
        r = minimize(soft_min_obj, cur.flatten(), args=(10.0,), method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 80, 'ftol': 1e-10})
        if -r.fun > min_area(cur):
            cur = clamp_bary(r.x.reshape(N, 2))
    except Exception:
        pass
    if time.time() - start > time_limit * 0.2:
        return cur
    
    # Stage 2: soft min beta=50
    try:
        r = minimize(soft_min_obj, cur.flatten(), args=(50.0,), method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 150, 'ftol': 1e-12})
        if -r.fun > min_area(cur):
            cur = clamp_bary(r.x.reshape(N, 2))
    except Exception:
        pass
    if time.time() - start > time_limit * 0.4:
        return cur
    
    # Stage 3: soft min beta=200
    try:
        r = minimize(soft_min_obj, cur.flatten(), args=(200.0,), method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14})
        if -r.fun > min_area(cur):
            cur = clamp_bary(r.x.reshape(N, 2))
    except Exception:
        pass
    if time.time() - start > time_limit * 0.6:
        return cur
    
    # Stage 4: NM on hard min
    try:
        r = minimize(hard_min_obj, cur.flatten(), method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
        cur = clamp_bary(r.x.reshape(N, 2))
    except Exception:
        pass
    
    return cur

def coordinate_descent(bary_init, time_limit=5.0):
    start = time.time()
    cur = bary_init.copy()
    best_val = min_area(cur)
    best_bary = cur.copy()
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start > time_limit:
                break
            pt = cur[i].copy()
            areas_all = calc_areas_bary(cur)
            mask_inv = (tri_idx[:, 0] == i) | (tri_idx[:, 1] == i) | (tri_idx[:, 2] == i)
            areas_not = areas_all[~mask_inv]
            min_not = np.min(areas_not) if len(areas_not) > 0 else np.inf
            
            def obj_i(x):
                bt = cur.copy()
                bt[i, 0] = x[0]
                bt[i, 1] = x[1]
                bt = clamp_bary(bt)
                areas_inv = calc_areas_bary(bt)[mask_inv]
                return -min(np.min(areas_inv), min_not)
            
            try:
                res = minimize(obj_i, pt, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13})
                new_pt = np.clip(res.x, 1e-10, 1 - 1e-10)
                if new_pt[0] + new_pt[1] > 1 - 1e-10:
                    s = (1 - 1e-10) / (new_pt[0] + new_pt[1])
                    new_pt *= s
                    new_pt = np.clip(new_pt, 1e-10, 1 - 1e-10)
                cur[i] = new_pt
                cur = clamp_bary(cur)
                new_val = min_area(cur)
                if new_val > best_val + 1e-13:
                    best_val = new_val
                    best_bary = cur.copy()
                    improved = True
            except Exception:
                pass
    return best_bary

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Symmetric template (high quality starting point)
    candidates.append(gen_sym_template())
    
    # Hexagonal
    candidates.append(gen_hex_init())
    
    # Random starts
    for i in range(6):
        candidates.append(gen_random(100 + i))
    
    # Perturbations of best symmetric
    sym = candidates[0]
    for i in range(4):
        candidates.append(gen_perturb(sym, 0.04, 200 + i))
    
    # Phase 2: Quick refine all
    refined = []
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        tlim = min(4.0, deadline - time.time() - 8)
        if tlim < 1.5:
            break
        try:
            r = refine_local(cand, time_limit=tlim)
            refined.append(r)
            update(r)
        except Exception:
            pass
    
    # Phase 3: Deep refine top candidates
    refined.sort(key=lambda b: min_area(b), reverse=True)
    top = refined[:6]
    
    for cand in top:
        if time.time() > deadline - 5:
            break
        tlim = min(12.0, deadline - time.time() - 3)
        if tlim < 2.0:
            break
        try:
            cd = coordinate_descent(cand, time_limit=tlim * 0.5)
            update(cd)
            rem = deadline - time.time() - 2
            if rem > 2.0:
                final = refine_local(cd, time_limit=rem)
                update(final)
        except Exception:
            pass
    
    # Phase 4: Final polish
    if best_bary is not None and time.time() < deadline - 2:
        rem = deadline - time.time() - 1
        if rem > 1.0:
            try:
                r = minimize(hard_min_obj, best_bary.flatten(), method='Nelder-Mead',
                            options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
                update(r.x.reshape(N, 2))
            except Exception:
                pass
    
    if best_bary is None:
        best_bary = gen_hex_init()
    
    return bary_to_cart(best_bary)
