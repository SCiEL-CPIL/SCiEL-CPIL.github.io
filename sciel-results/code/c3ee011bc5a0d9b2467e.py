import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = 2.0 * y / H
    u = 1.0 - v - x
    return np.column_stack([u, v])

def clamp_bary(bary, eps=1e-12):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
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

def get_areas_cartesian(cart):
    """Compute all triangle areas."""
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    return 0.5 * np.abs(cross)

def calc_min_area_cartesian(points):
    """Calculate the minimum area of all triangles formed by N points."""
    areas = get_areas_cartesian(points)
    return np.min(areas)

def epigraph_obj(z, beta=50.0, lambda_pen=10.0):
    """
    Smooth epigraph objective.
    z: [bary_1, bary_2, ..., bary_N, t]
    Minimize -t + lambda * sum(softplus(beta * (t - area_i)))
    """
    bary = z[:N*2].reshape(N, 2)
    t = z[-1]
    
    # Ensure bary is valid for area calculation (clamp for safety, though bounds should handle it)
    bary_clamped = clamp_bary(bary)
    cart = bary_to_cart(bary_clamped)
    areas = get_areas_cartesian(cart)
    
    # softplus(x) = log(1 + exp(x))
    # We want to penalize (t - area) if t > area
    # diff = t - areas
    diff = t - areas
    # Clip diff to avoid overflow in exp
    diff_clipped = np.clip(diff, -50, 50)
    sp = np.logaddexp(0, beta * diff_clipped)
    
    penalty = lambda_pen * np.sum(sp)
    obj = -t + penalty
    
    return obj

def hard_min_obj_bary(bary_flat):
    """Objective using hard min in barycentric coords."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    return -calc_min_area_cartesian(cart)

def generate_lattice_init():
    """Generate a triangular lattice initialization."""
    pts = []
    for i in range(5):
        for j in range(5 - i):
            pts.append([i / 4.0, j / 4.0])
    pts = np.array(pts)
    
    center = np.array([1/3, 1/3])
    pts = np.vstack([pts, center])
    
    return pts

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_epigraph(bary_init, t_init=0.01, time_limit=5.0):
    """
    Refine using graduated epigraph optimization.
    """
    start_time = time.time()
    bary = bary_init.copy()
    t = t_init
    
    # Sequence of betas for gradual sharpening
    betas = [5.0, 20.0, 100.0, 500.0]
    lambdas = [10.0, 10.0, 5.0, 1.0]
    
    for idx, beta in enumerate(betas):
        if time.time() - start_time > time_limit - 1:
            break
            
        lambda_pen = lambdas[idx]
        
        # Initial guess for z: [bary_flat, t]
        z0 = np.concatenate([bary.flatten(), [t]])
        
        # Bounds: u in [0,1], v in [0,1], t in [0, 0.1]
        bounds = [(0, 1)] * (N * 2) + [(0, 0.1)]
        
        try:
            res = minimize(
                epigraph_obj,
                z0,
                method='SLSQP',
                bounds=bounds,
                args=(beta, lambda_pen),
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or (res.fun < epigraph_obj(z0, beta, lambda_pen)):
                z_opt = res.x
                bary = clamp_bary(z_opt[:N*2].reshape(N, 2))
                t = z_opt[-1]
        except Exception:
            pass
            
    return bary, t

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary, val=None):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        if val is None:
            val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Lattice Init
    lat_init = generate_lattice_init()
    bary_ref, t_ref = refine_epigraph(lat_init, t_init=0.005, time_limit=10.0)
    update_best(bary_ref)
    
    # 2. Random Restarts
    rng = np.random.default_rng(42)
    
    # We will do several random restarts and some perturbed best restarts
    num_random_starts = 15
    
    for i in range(num_random_starts):
        if time.time() > deadline - 20:
            break
            
        init_bary = generate_random_init(seed=100 + i)
        # Use a small t_init
        bary_ref, t_ref = refine_epigraph(init_bary, t_init=0.001, time_limit=4.0)
        update_best(bary_ref)
        
        # Occasionally perturb the best
        if i % 3 == 0 and best_bary is not None and time.time() < deadline - 30:
            noise = rng.normal(0, 0.05, best_bary.shape)
            perturbed = clamp_bary(best_bary + noise)
            bary_ref, t_ref = refine_epigraph(perturbed, t_init=best_val, time_limit=4.0)
            update_best(bary_ref)

    # 3. Final High-Precision Polish
    if best_bary is not None and time.time() < deadline - 5:
        # Use Nelder-Mead on the hard min for final tweak
        try:
            res_nm = minimize(
                hard_min_obj_bary,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < -best_val: # i.e., -min_area < -best_val => min_area > best_val
                update_best(clamp_bary(res_nm.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One more SLSQP with very high beta
        if time.time() < deadline - 2:
            try:
                t_init = best_val
                z0 = np.concatenate([best_bary.flatten(), [t_init]])
                bounds = [(0, 1)] * (N * 2) + [(0, 0.1)]
                res_final = minimize(
                    epigraph_obj,
                    z0,
                    method='SLSQP',
                    bounds=bounds,
                    args=(1000.0, 0.1),
                    options={'maxiter': 1000, 'ftol': 1e-14}
                )
                if res_final.fun < epigraph_obj(z0, 1000.0, 0.1):
                    bary_opt = clamp_bary(res_final.x[:N*2].reshape(N, 2))
                    update_best(bary_opt)
            except Exception:
                pass

    cart = bary_to_cart(best_bary)
    return cart
