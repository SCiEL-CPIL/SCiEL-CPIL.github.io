"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Strategy: 
1. Use a smooth objective function: sum of inverse squares of triangle areas.
   This penalizes small triangles heavily and is differentiable everywhere.
2. Use L-BFGS-B for fast local convergence on this smooth landscape.
3. Use Nelder-Mead for final polish on the hard min objective to squeeze out last bits of precision.
4. Multi-start with diverse initializations (Random, Hexagonal, Repulsive).
"""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute triangle indices
tri_idx = np.array(list(combinations(range(N), 3)))  # (165, 3)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    w = 1 - u - v
    Point = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w * H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios (relative to big triangle area).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-8):
    """
    Clamp barycentric coordinates to the valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """
    Objective function using hard min. Returns negative min area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def smooth_obj(bary_flat, eps=1e-6):
    """
    Smooth objective: Sum of inverse squares of areas.
    Minimization of this drives the smallest area up.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Avoid division by zero
    safe_areas = np.maximum(areas, eps)
    return np.sum(1.0 / (safe_areas ** 2))

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration in barycentric coords.
    """
    best_bary = None
    best_val = np.inf
    
    # Try different scales
    for s in np.linspace(0.1, 0.4, 15):
        points = []
        for i in range(10):
            for j in range(10):
                u = i * s
                v = j * s
                if u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = smooth_obj(bary.flatten())
            if val < best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        best_bary = generate_random_init(0)
        
    return best_bary

def generate_repulsive_init(seed=None):
    """
    Generate initial points using a simple repulsive force simulation.
    """
    rng = np.random.default_rng(seed)
    bary = generate_random_init(seed)
    
    # Simple repulsion steps
    for _ in range(500):
        forces = np.zeros_like(bary)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                diff = bary[i] - bary[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    forces[i] += rng.normal(0, 0.01, 2)
                else:
                    forces[i] += diff / (dist_sq ** 1.5) # 1/r^2 force in 2D barycentric approx
        
        bary += 0.01 * forces
        bary = clamp_bary(bary)
        
    return bary

def try_optimize_smooth(bary_init, deadline=None):
    """
    Optimize using the smooth inverse-square objective with L-BFGS-B.
    """
    if deadline is not None and time.time() > deadline - 10:
        return bary_init
    
    try:
        bounds = [(0, 1)] * (N * 2)
        
        result = minimize(
            smooth_obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        
        return result.x.reshape(N, 2)
        
    except Exception:
        return bary_init

def nelder_mead_polish(bary_init, deadline=None, maxiter=1000):
    """
    Polish a configuration using Nelder-Mead on the hard min objective.
    """
    if deadline is not None and time.time() > deadline - 5.0:
        return bary_init
    
    try:
        res_nm = minimize(
            hard_min_obj,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res_nm.x.reshape(N, 2)
    except Exception:
        return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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

    # 1. Hexagonal Initialization
    hex_init = generate_hexagonal_init()
    result = try_optimize_smooth(hex_init, deadline=deadline)
    update_best(result)
    
    # 2. Repulsive Initialization
    if time.time() < deadline - 30:
        rep_init = generate_repulsive_init(seed=1)
        result = try_optimize_smooth(rep_init, deadline=deadline)
        update_best(result)
        
    # 3. Multiple Random Initializations
    num_random_starts = 10
    for i in range(num_random_starts):
        if time.time() > deadline - 40:
            break
        rand_init = generate_random_init(seed=100 + i)
        result = try_optimize_smooth(rand_init, deadline=deadline)
        update_best(result)
        
    # 4. Perturb Best and Re-optimize (Local Search around Best)
    if best_bary is not None:
        for i in range(5):
            if time.time() > deadline - 20:
                break
            rng = np.random.default_rng(200 + i)
            perturbed = best_bary + rng.normal(0, 0.05, best_bary.shape)
            perturbed = clamp_bary(perturbed)
            result = try_optimize_smooth(perturbed, deadline=deadline)
            update_best(result)
            
    # 5. Final Polish with Nelder-Mead on Hard Min
    if best_bary is not None and time.time() < deadline - 5.0:
        nm_result = nelder_mead_polish(best_bary, deadline=deadline, maxiter=2000)
        update_best(nm_result)
        
    # Fallback
    if best_bary is None:
        best_bary = generate_random_init(seed=0)
        
    # Final Safety Clamp and Convert to Cartesian
    best_bary = clamp_bary(best_bary)
    result_cart = bary_to_cart(best_bary)
    
    # Ensure all points are strictly inside
    result_cart = np.clip(result_cart, 1e-9, None)
    result_cart[:, 1] = np.clip(result_cart[:, 1], 1e-9, H - 1e-9)
    x_min = result_cart[:, 1] / np.sqrt(3)
    x_max = 1.0 - result_cart[:, 1] / np.sqrt(3)
    result_cart[:, 0] = np.clip(result_cart[:, 0], x_min + 1e-9, x_max - 1e-9)
    
    return result_cart
