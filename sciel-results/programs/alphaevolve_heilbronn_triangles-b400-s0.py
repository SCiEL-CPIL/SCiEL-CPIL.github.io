"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

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
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0)
    v = np.clip(v, eps, 1.0)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.maximum(sums[invalid], eps)
        u[invalid] *= scale
        v[invalid] *= scale
        
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
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def power_sum_obj(bary_flat, p=8.0):
    """
    Objective based on sum of inverse powers of areas.
    This emphasizes small areas more strongly.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    # Use a floor to avoid division by zero
    areas_safe = np.maximum(areas, 1e-10)
    return np.sum(areas_safe ** (-p))

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.45, 30):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s * 0.866  # sqrt(3)/2
                if u + v <= 1.0 and len(points) < N:
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

def generate_perturbed_init(base_bary, sigma=0.08, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    # Stage 1: Smooth soft-min minimization with L-BFGS-B
    try:
        result1 = minimize(
            soft_min_obj,
            bary_init.flatten(),
            args=(80.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
    except Exception:
        result1 = type('obj', (), {'x': bary_init.flatten()})()
    
    if time.time() - start_time > time_limit * 0.5:
        return clamp_bary(result1.x.reshape(N, 2))
        
    bary_mid = clamp_bary(result1.x.reshape(N, 2))
    
    # Stage 2: Power sum refinement
    try:
        result2 = minimize(
            power_sum_obj,
            bary_mid.flatten(),
            args=(8.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
    except Exception:
        result2 = type('obj', (), {'x': bary_mid.flatten()})()
    
    if time.time() - start_time > time_limit * 0.8:
        return clamp_bary(result2.x.reshape(N, 2))
    
    bary_mid2 = clamp_bary(result2.x.reshape(N, 2))
    
    # Stage 3: Hard min refinement with Nelder-Mead
    try:
        result3 = minimize(
            hard_min_obj,
            bary_mid2.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-16}
        )
    except Exception:
        return bary_mid2
        
    return clamp_bary(result3.x.reshape(N, 2))

def anneal(bary_init, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_val = -hard_min_obj(bary.flatten())
    best_bary = bary.copy()
    best_val = current_val
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.995
    max_iters = 2000
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Generate a random perturbation
        sigma = T * 2.0
        noise = np.random.normal(0, sigma, bary.shape)
        bary_new = clamp_bary(bary + noise)
        
        new_val = -hard_min_obj(bary_new.flatten())
        delta = new_val - current_val
        
        # Accept or reject
        if delta > 0 or np.random.random() < np.exp(delta / T):
            bary = bary_new
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_bary = bary.copy()
        
        T = max(T * cooling_rate, T_min)
    
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
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
    
    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=6.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 50:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=6.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Simulated annealing from best
    if best_bary is not None and time.time() < deadline - 30:
        annealed = anneal(best_bary, time_limit=15.0)
        refined_annealed = refine_local(annealed, time_limit=6.0)
        update_best(refined_annealed)
    
    # 4. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.05, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)
        
        # Occasionally try annealing from a random start
        if restart_count % 5 == 0 and time.time() < deadline - 25:
            annealed = anneal(init, time_limit=5.0)
            refined_annealed = refine_local(annealed, time_limit=3.0)
            update_best(refined_annealed)
    
    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-18}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
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
