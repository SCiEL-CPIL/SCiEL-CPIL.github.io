"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
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

def clamp_bary(bary, eps=1e-6):
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

def soft_min_obj(bary_flat, beta=100.0):
    """Objective: minimize negative soft-min of areas."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
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

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    
    def try_two_stage(bary_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        """Two-stage optimization: smooth then sharp landscape."""
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1: smooth landscape
            def obj1(x):
                return soft_min_obj(x, beta=beta1)
            
            bounds = [(0, 1)] * (N * 2)
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 10:
                return
            
            # Stage 2: sharp landscape
            def obj2(x):
                return soft_min_obj(x, beta=beta2)
            
            result2 = minimize(
                obj2,
                result1.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            final_bary = clamp_bary(result2.x.reshape(N, 2))
            update_best(final_bary)
            
        except Exception:
            pass
    
    def try_de_global():
        """Differential evolution for global search."""
        if time.time() > deadline - 60:
            return
        
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            result = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-8,
                polish=True
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    # 1. Start with hexagonal lattice
    hex_init = generate_hexagonal_init()
    try_two_stage(hex_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300)
    
    # 2. Global search with DE
    try_de_global()
    
    # 3. Multiple random starts
    for i in range(10):
        if time.time() > deadline - 30:
            break
        rand_init = generate_random_init(seed=100 + i)
        try_two_stage(rand_init, beta1=40.0, beta2=150.0, maxiter1=150, maxiter2=250)
    
    # 4. Perturb best and re-optimize
    if best_bary is not None:
        for i in range(6):
            if time.time() > deadline - 20:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            try_two_stage(perturbed, beta1=80.0, beta2=300.0, maxiter1=200, maxiter2=300)
    
    # 5. Nelder-Mead polish on hard min
    if best_bary is not None and time.time() < deadline - 10:
        try:
            bary_nm = best_bary.copy()
            bary_nm[:, 0] = np.clip(bary_nm[:, 0], 1e-6, 1 - 1e-6)
            bary_nm[:, 1] = np.clip(bary_nm[:, 1], 1e-6, 1 - bary_nm[:, 0] - 1e-6)
            
            result = minimize(
                hard_min_obj,
                bary_nm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16}
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    # 6. Final small perturbations
    if best_bary is not None and time.time() < deadline - 5:
        for i in range(15):
            if time.time() > deadline - 3:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.005, seed=300 + i)
            update_best(perturbed)
    
    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    # Safety check
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
    
    return cart
