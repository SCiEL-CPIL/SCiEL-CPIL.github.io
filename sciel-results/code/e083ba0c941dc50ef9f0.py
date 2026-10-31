import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def clamp_bary(bary, eps=1e-9):
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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
        
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def local_refine(bary_init, time_limit=5.0):
    """
    Refine a configuration using gradient-based steps on a smooth surrogate,
    followed by Nelder-Mead on the hard min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    bary_current = bary_init.copy()
    
    # Stage 1: L-BFGS-B on soft min (smooths the landscape for gradients)
    # Use a high beta to approximate the min closely
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.5:
        return bary_current

    # Stage 2: Nelder-Mead on hard min (polish)
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def simulated_annealing(bary_init, time_limit=10.0):
    """
    Run Simulated Annealing to escape local minima.
    """
    start_time = time.time()
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    
    best_bary = bary_current.copy()
    best_val = current_val
    
    steps = 0
    while time.time() - start_time < time_limit:
        # Propose a move: pick a random point and move it slightly
        idx = np.random.randint(N)
        # Random perturbation in barycentric space
        du = np.random.normal(0, 0.05)
        dv = np.random.normal(0, 0.05)
        
        bary_new = bary_current.copy()
        bary_new[idx, 0] += du
        bary_new[idx, 1] += dv
        
        bary_new = clamp_bary(bary_new)
        
        new_val = -hard_min_obj(bary_new.flatten())
        delta = new_val - current_val
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            bary_current = bary_new
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_bary = bary_current.copy()
                
        T *= cooling_rate
        steps += 1
        
        # Occasionally do a local refinement to push the best found so far
        if steps % 100 == 0 and time.time() - start_time < time_limit - 5:
            refined = local_refine(best_bary, time_limit=2.0)
            refined_val = -hard_min_obj(refined.flatten())
            if refined_val > best_val:
                best_val = refined_val
                best_bary = refined.copy()
                bary_current = best_bary.copy()
                current_val = best_val
                # Reset temperature slightly to explore around the new best
                T = max(T, 0.005)

    return best_bary

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

    # 1. Generate diverse initial configurations
    # Use a mix of random and structured starts
    rng = np.random.default_rng(42)
    
    inits = []
    
    # Random start
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    inits.append(np.column_stack([u, v]))
    
    # Jittered Grid Start
    # Create a coarse grid in barycentric coords
    grid_size = 4
    us = np.linspace(0.1, 0.8, grid_size)
    vs = np.linspace(0.1, 0.8, grid_size)
    grid_pts = []
    for u in us:
        for v in vs:
            if u + v < 0.9:
                grid_pts.append([u, v])
    if len(grid_pts) >= N:
        # Pick N points from grid with some jitter
        indices = rng.choice(len(grid_pts), N, replace=False)
        grid_init = np.array(grid_pts)[indices]
        grid_init += rng.normal(0, 0.05, grid_init.shape)
        inits.append(clamp_bary(grid_init))
        
    # Hexagonal-like packing (approx)
    # Points near vertices and center
    hex_init = np.array([
        [0.05, 0.05], [0.90, 0.05], [0.05, 0.90], # Near vertices (bary)
        [0.5, 0.05], [0.05, 0.5], [0.5, 0.5], # Edge midpoints/center
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], # Inner
        [0.5, 0.3], [0.3, 0.5]
    ])
    hex_init += rng.normal(0, 0.02, hex_init.shape)
    inits.append(clamp_bary(hex_init))

    # 2. Run SA on each init
    # Allocate time based on number of inits
    time_per_init = (deadline - start_time - 10) / len(inits)
    
    for i, init_bary in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Run SA
        sa_result = simulated_annealing(init_bary, time_limit=min(time_per_init, 30.0))
        update_best(sa_result)
        
        # Final local refine on the best from SA
        if time.time() < deadline - 10:
            refined = local_refine(best_bary, time_limit=min(10.0, deadline - 10 - time.time()))
            update_best(refined)

    # 3. Final Global Polish
    if best_bary is not None and time.time() < deadline - 5:
        final_polish = local_refine(best_bary, time_limit=min(5.0, deadline - 5 - time.time()))
        update_best(final_polish)

    if best_bary is None:
        # Absolute fallback
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    cart = bary_to_cart(best_bary)
    return cart

if __name__ == "__main__":
    pts = construct()
    print(pts)
