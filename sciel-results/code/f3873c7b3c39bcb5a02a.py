import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
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
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    
    # Area is proportional to the determinant. 
    # In barycentric coords (u,v) where w=1-u-v, the determinant of 
    # [[u_i, v_i, w_i], ...] is the same as [[u_i, v_i, 1], ...] 
    # up to a constant factor? 
    # Actually, let's stick to the robust cross product calculation in 2D barycentric space 
    # or just use the standard determinant of the 3x3 matrix with a column of 1s.
    # The area of triangle (P1, P2, P3) in Cartesian is 0.5 * |det(P1, P2, P3)|.
    # In barycentric coordinates relative to the unit area triangle?
    # The reference triangle has area sqrt(3)/4.
    # If we use barycentric coordinates (u, v) such that P = u*V0 + v*V1 + (1-u-v)*V2.
    # The determinant of the matrix with rows [u_i, v_i, 1] gives the ratio of the area 
    # of the triangle formed by the points to the area of the reference triangle.
    # So we can just compute abs(det) of [[u, v, 1], ...].
    
    # Let's verify the formula used in the current program:
    # det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    # This is the determinant of [[ui, vi, wi], [uj, vj, wj], [uk, vk, wk]].
    # Since ui+vi+wi = 1, this determinant is equal to the determinant of [[ui, vi, 1], ...].
    # And this determinant represents the area of the triangle formed by the points 
    # divided by the area of the reference triangle.
    # Since the reference triangle is unit-area (in the problem statement "unit-area triangle"),
    # this value is exactly the area we want to maximize.
    
    # Using the cross product form for speed:
    # det = (uj-ui)*(vk-vi) - (vj-vi)*(uk-ui) is NOT correct for barycentric directly 
    # unless we transform. 
    # Let's stick to the full determinant or the explicit barycentric area formula.
    # The formula in the current program is correct for the ratio.
    
    # However, the inspiration used:
    # cross = (uj - ui) * (vk - vi) - (vj - vi) * (uk - ui)
    # This is the 2D cross product of vectors (Pj-Pi) and (Pk-Pi) in the (u,v) plane?
    # No, (u,v) are coordinates in the plane of the triangle.
    # The area of the triangle in Cartesian is proportional to the area in the (u,v) coordinate system 
    # scaled by the area of the reference triangle.
    # Specifically, Area_cart = Area_ref * |det_bary|.
    # Since Area_ref = 1 (unit area triangle), Area_cart = |det_bary|.
    
    # The determinant of [[u,v,w]] is equal to the determinant of [[u,v,1]].
    # det([[u1, v1, 1], [u2, v2, 1], [u3, v3, 1]])
    # = u1(v2-v3) - v1(u2-u3) + (u2v3 - u3v2)
    
    # Let's use this explicit formula for clarity and speed.
    det = ui * (vj - vk) - vi * (uj - uk) + (uj * vk - uk * vj)
    return np.abs(det)

def clamp_bary(bary, eps=1e-10):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def epigraph_obj(x):
    """
    Epigraph formulation: maximize t subject to areas >= t.
    x = [bary_flat (22), t]
    Minimize -t + penalty * sum(max(0, t - areas)^2)
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalize violations of areas >= t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations**2)
    
    # Large penalty coefficient to enforce constraints
    return -t + 1000.0 * penalty

def generate_hexagonal_init():
    best_bary = None
    best_val = -np.inf
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        if len(pts) >= N:
            rng = np.random.default_rng(42)
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts, dtype=float)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts, dtype=float)[idx]
                bary = clamp_bary(bary)
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = clamp_bary(np.column_stack([u, v]))
    return best_bary

def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return clamp_bary(np.column_stack([u, v]))

def generate_perturbed_init(base_bary, sigma=0.04, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def generate_boundary_interior_init(seed=None):
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    bary[0] = [0.08, 0.08]
    bary[1] = [0.92, 0.08]
    bary[2] = [0.08, 0.92]
    for i in range(3, 7):
        edge = (i - 3) % 3
        t = rng.uniform(0.25, 0.75)
        if edge == 0:
            bary[i] = [t, 0.03]
        elif edge == 1:
            bary[i] = [0.03, t]
        else:
            bary[i] = [1.0 - t - 0.03, 0.03]
    for i in range(7, 11):
        u = rng.uniform(0.15, 0.75)
        v = rng.uniform(0.15, 0.75 * (1 - u))
        bary[i] = [u, v]
    return clamp_bary(bary)

def refine_local(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(1e-9, 1 - 1e-9)] * (N * 2)
    bary_current = bary_init.copy()
    
    # Stage 1: Epigraph formulation with SLSQP
    # This is the structural change. It handles the min-area constraint smoothly.
    try:
        # Initial t is the current min area
        areas = calc_areas_bary(bary_current)
        t0 = np.min(areas)
        
        # Bounds for t: [0, 1] (max possible area is 1 for unit triangle)
        epigraph_bounds = bounds + [(0.0, 1.0)]
        
        result1 = minimize(
            epigraph_obj, 
            np.concatenate([bary_current.flatten(), [t0]]),
            method='SLSQP', 
            bounds=epigraph_bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        # Extract new bary and t
        new_bary = clamp_bary(result1.x[:N*2].reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        
        if new_val > -hard_min_obj(bary_current.flatten()):
            bary_current = new_bary
    except Exception:
        pass

    if time.time() - start_time > time_limit * 0.6:
        return bary_current

    # Stage 2: Nelder-Mead on hard min for final polish
    try:
        result2 = minimize(
            hard_min_obj, bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass

    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Hexagonal init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)

    # 2. Boundary-interior init
    bi_init = generate_boundary_interior_init(seed=123)
    refined_bi = refine_local(bi_init, time_limit=8.0)
    update_best(refined_bi)

    # 3. Differential Evolution
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            bounds = [(1e-9, 1 - 1e-9)] * (N * 2)
            result_de = differential_evolution(
                de_obj, bounds, seed=42,
                maxiter=30, popsize=12, tol=1e-6, polish=False
            )
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass

    # 4. Multi-start with varied initializations
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            init = generate_boundary_interior_init(seed=4000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)

        refined = refine_local(init, time_limit=5.0)
        update_best(refined)

    # 5. Final high-precision polish using Epigraph again if time permits
    if best_bary is not None and time.time() < deadline - 5:
        try:
            areas = calc_areas_bary(best_bary)
            t0 = np.min(areas)
            epigraph_bounds = [(1e-9, 1 - 1e-9)] * (N * 2) + [(0.0, 1.0)]
            result_final = minimize(
                epigraph_obj, 
                np.concatenate([best_bary.flatten(), [t0]]),
                method='SLSQP', 
                bounds=epigraph_bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            new_bary = clamp_bary(result_final.x[:N*2].reshape(N, 2))
            update_best(new_bary)
        except Exception:
            pass

    if best_bary is None:
        best_bary = generate_random_init(seed=0)

    cart = bary_to_cart(best_bary)
    return cart
