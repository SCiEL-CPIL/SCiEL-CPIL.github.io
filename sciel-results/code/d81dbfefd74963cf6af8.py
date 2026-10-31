"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

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

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
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

def clamp_bary(bary, eps=1e-7):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective: negative of minimum triangle area."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """
    Soft minimum using log-sum-exp for smoothness.
    Minimizing this encourages all areas to be large, with emphasis on the smallest.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    # We want to maximize the minimum, so we minimize -soft_min
    # soft_min = -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    shifted = -beta * areas
    m = np.max(shifted)
    log_sum = m + np.log(np.sum(np.exp(shifted - m)))
    soft_min_val = -log_sum / beta
    return -soft_min_val

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.45, 30):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s * 0.866  # approx sqrt(3)/2
                if u + v <= 0.98 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def generate_perturbed_init(base_bary, noise_scale=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_scale, base_bary.shape)
    perturbed = base_bary + noise
    return clamp_bary(perturbed)

def refine_local(bary_init, time_limit=10.0, start_time=None):
    """
    Refine a configuration using a sequence of optimizers with increasing smoothness control.
    """
    if start_time is None:
        start_time = time.time()
        
    bounds = [(0, 1)] * (N * 2)
    current_bary = bary_init.copy()
    
    # Stage 1: Soft min with moderate beta
    try:
        if time.time() - start_time < time_limit * 0.4:
            res1 = minimize(
                soft_min_obj,
                current_bary.flatten(),
                args=(20.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            if res1.success or res1.fun < -hard_min_obj(current_bary.flatten()):
                current_bary = clamp_bary(res1.x.reshape(N, 2))
    except Exception:
        pass
        
    # Stage 2: Soft min with higher beta (sharper)
    try:
        if time.time() - start_time < time_limit * 0.7:
            res2 = minimize(
                soft_min_obj,
                current_bary.flatten(),
                args=(100.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            if res2.success or res2.fun < -hard_min_obj(current_bary.flatten()):
                current_bary = clamp_bary(res2.x.reshape(N, 2))
    except Exception:
        pass
        
    # Stage 3: Hard min with Nelder-Mead (unconstrained, relies on clamping)
    try:
        if time.time() - start_time < time_limit * 0.9:
            res3 = minimize(
                hard_min_obj,
                current_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res3.fun < -hard_min_obj(current_bary.flatten()):
                current_bary = clamp_bary(res3.x.reshape(N, 2))
    except Exception:
        pass
        
    return current_bary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
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

    # 1. Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=15.0, start_time=start_time)
    update_best(refined_hex)
    
    # 2. Random Restarts with Local Refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    noise_scales = [0.1, 0.05, 0.02, 0.01]
    
    while time.time() < deadline - 5:
        restart_count += 1
        
        # Alternate between random init and perturbation of best
        if best_bary is not None and restart_count % 3 != 0:
            # Perturb best
            scale = noise_scales[restart_count % len(noise_scales)]
            init = generate_perturbed_init(best_bary, noise_scale=scale, seed=1000 + restart_count)
        else:
            # Random init
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init = np.column_stack([u, v])
        
        # Time limit per restart decreases as we go
        time_left = deadline - time.time() - 5
        if time_left < 3:
            break
            
        t_limit = min(10.0, time_left * 0.5)
        refined = refine_local(init, time_limit=t_limit, start_time=start_time)
        update_best(refined)
        
    # 3. Final Polish
    if best_bary is not None and time.time() < deadline - 1:
        try:
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass

    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    bary_final = clamp_bary(best_bary, eps=1e-6)
    cart = bary_to_cart(bary_final)
    
    # Safety check to ensure points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
    
    return cart

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

# Ensure construct is the entry point
if __name__ == "__main__":
    result = construct()
    print(result)
