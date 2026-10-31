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

def soft_min_obj(bary_flat, beta=50.0):
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
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
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
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def refine_local(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(10.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.2:
        return bary_current

    # Stage 2: Medium soft-min
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.4:
        return bary_current

    # Stage 3: Sharp soft-min
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result3.success or result3.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.6:
        return bary_current

    # Stage 4: Nelder-Mead on Hard Min
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final Polish NM
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def simulated_annealing(bary_init, time_limit=10.0, rng_seed=42):
    """
    Simulated Annealing to maximize the minimum triangle area.
    Uses soft-min energy to guide the search, annealing beta to sharpen.
    """
    start_time = time.time()
    rng = np.random.default_rng(rng_seed)
    
    current_bary = bary_init.copy()
    current_bary = clamp_bary(current_bary)
    
    # Energy function: -soft_min(areas, beta)
    # We want to minimize this (i.e., maximize soft_min)
    
    # Initial temperature
    T = 0.01
    T_min = 1e-5
    cooling_rate = 0.95
    
    # Initial beta for soft min
    beta = 10.0
    beta_max = 100.0
    
    # Evaluate initial
    current_areas = calc_areas_bary(current_bary)
    current_min_area = np.min(current_areas)
    
    # Energy: we use -min(areas) for hard acceptance if beta is high, 
    # but for SA we want a smooth energy.
    # Let's define energy E = -soft_min(areas, beta)
    
    def get_energy(bary, beta_val):
        areas = calc_areas_bary(bary)
        if np.any(areas < 1e-15):
            return 1e10
        scaled = -beta_val * areas
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            return -np.min(areas)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta_val
        return -soft_min

    current_energy = get_energy(current_bary, beta)
    best_bary = current_bary.copy()
    best_energy = current_energy
    best_val = -np.min(calc_areas_bary(best_bary))

    step_size = 0.05

    while time.time() - start_time < time_limit:
        # Propose a move
        # Perturb 1 or 2 random points
        num_perturb = rng.integers(1, 3)
        indices = rng.choice(N, num_perturb, replace=False)
        
        new_bary = current_bary.copy()
        for idx in indices:
            noise = rng.normal(0, step_size, 2)
            new_bary[idx] += noise
            
        new_bary = clamp_bary(new_bary)
        
        # Calculate new energy
        new_energy = get_energy(new_bary, beta)
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0:
            accept = True
        else:
            # P = exp(-delta_e / T)
            # If T is small, this is very small
            if T > 1e-9:
                prob = np.exp(-delta_e / T)
            else:
                prob = 0.0
            accept = rng.random() < prob
            
        if accept:
            current_bary = new_bary
            current_energy = new_energy
            
            # Update best
            val = -np.min(calc_areas_bary(current_bary))
            if val > best_val:
                best_val = val
                best_energy = current_energy
                best_bary = current_bary.copy()
                
        # Anneal
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optionally restart or reduce step size
            step_size *= 0.9
            if step_size < 1e-4:
                break
                
        # Increase beta slowly to sharpen the minimum
        beta = min(beta * 1.01, beta_max)

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

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Hexagonal
    candidates.append(generate_hexagonal_init())
    
    # 2. Random starts
    for i in range(5):
        candidates.append(generate_random_init(seed=100 + i))

    # Phase 2: Simulated Annealing on each candidate
    # Allocate time based on remaining budget
    num_candidates = len(candidates)
    sa_time_per_candidate = min(10.0, (TIME_LIMIT - 20.0) / num_candidates)
    
    sa_results = []
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        try:
            sa_refined = simulated_annealing(cand, time_limit=sa_time_per_candidate, rng_seed=1000 + i)
            sa_results.append(sa_refined)
            update_best(sa_refined)
        except Exception:
            pass

    # Phase 3: Deep Refinement on Top SA Results
    def get_val(b):
        return -hard_min_obj(b.flatten())
        
    sa_results.sort(key=get_val, reverse=True)
    
    # Take top 3 for deeper refinement
    top_candidates = sa_results[:3]
    
    for i, cand in enumerate(top_candidates):
        if time.time() > deadline - 5:
            break
        
        t_limit = min(10.0, deadline - time.time() - 2)
        if t_limit < 2.0:
            break
            
        try:
            final_refined = refine_local(cand, time_limit=t_limit)
            update_best(final_refined)
        except Exception:
            pass

    # Phase 4: Final Polish on the absolute best
    if best_bary is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    if best_bary is None:
        best_bary = generate_hexagonal_init()

    cart = bary_to_cart(best_bary)
    return cart
