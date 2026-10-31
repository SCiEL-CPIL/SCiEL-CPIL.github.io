import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
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

def epigraph_obj(x, beta=100.0):
    """
    Epigraph formulation: x = [bary_flat (22), t (1)]
    Minimize -t subject to soft constraints: all areas >= t (softly).
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas < t
    diff = areas - t
    if np.any(diff < -1e-10):
        penalty = np.sum(np.maximum(0, -diff) ** 2) * 1e6
        return -t + penalty
    
    # Soft constraint via log-sum-exp
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * diff
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -t + 1e6 * np.sum(np.maximum(0, -diff) ** 2)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    penalty = max(0, log_sum / beta) * 1e4
    return -t + penalty

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
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
            # Try different subsets
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
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def de_wrapper(x_flat, bounds_list):
    """Wrapper for differential_evolution that works with flat barycentric coords."""
    bary = x_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    return -hard_min_obj(bary.flatten())

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    5-stage approach: coarse -> medium -> sharp -> epigraph -> NM polish
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coarse soft-min (low beta, more iterations)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(20.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.25:
        return bary_current

    # Stage 2: Medium soft-min (increasing beta)
    try:
        result2 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.45:
        return bary_current

    # Stage 3: Sharp soft-min (high beta, close to hard min)
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
        
    if time.time() - start_time > time_limit * 0.65:
        return bary_current

    # Stage 4: Epigraph formulation with auxiliary variable t
    try:
        t_init = np.min(calc_areas_bary(bary_current))
        x_init = np.concatenate([bary_current.flatten(), [t_init]])
        bounds_epigraph = [(0, 1)] * (N * 2) + [(0, 1)]
        
        result4 = minimize(
            epigraph_obj,
            x_init,
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds_epigraph,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        new_bary = clamp_bary(result4.x[:2*N].reshape(N, 2))
        new_t = result4.x[2*N]
        
        # Check if the epigraph solution is valid and better
        new_areas = calc_areas_bary(new_bary)
        if np.min(new_areas) >= new_t - 1e-8 and np.min(new_areas) > -hard_min_obj(bary_current.flatten()):
            bary_current = new_bary
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Nelder-Mead on hard min for final polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        if result5.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass

    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_score = -np.inf
    best_points_bary = None
    
    # 1. Generate initial candidates
    candidates = []
    
    # Hexagonal init
    candidates.append(generate_hexagonal_init())
    
    # Random inits
    for i in range(8):
        candidates.append(generate_random_init(seed=i))
    
    # 2. Use differential evolution for global search
    de_time_budget = min(30.0, (deadline - start_time) * 0.25)
    if de_time_budget > 5:
        try:
            bounds_de = [(0, 1)] * (N * 2)
            
            def de_objective(x_flat):
                bary = x_flat.reshape(N, 2)
                bary = clamp_bary(bary)
                return -hard_min_obj(bary.flatten())
            
            de_result = differential_evolution(
                de_objective,
                bounds_de,
                maxiter=50,
                popsize=30,
                tol=1e-10,
                seed=42,
                polish=False,
                update_interval=1,
                callback=lambda x, convergence: None  # No callback to save time
            )
            
            de_bary = clamp_bary(de_result.x.reshape(N, 2))
            de_score = -hard_min_obj(de_bary.flatten())
            
            if de_score > best_score:
                best_score = de_score
                best_points_bary = de_bary.copy()
            
            # Add DE result as a candidate for local refinement
            candidates.append(de_bary)
            
        except Exception:
            pass
    
    # 3. Refine each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Time budget per candidate: distribute remaining time
        remaining = deadline - time.time()
        time_per_candidate = max(0.5, (remaining - 5.0) / max(1, len(candidates) - i))
        
        refined = refine_local(cand, time_limit=time_per_candidate)
        
        score = -hard_min_obj(refined.flatten())
        
        if score > best_score:
            best_score = score
            best_points_bary = refined.copy()
    
    # 4. If we have time left, do a few more random restarts from the best point
    if best_points_bary is not None and time.time() < deadline - 5:
        for i in range(5):
            if time.time() > deadline - 2:
                break
            perturbed = generate_perturbed_init(best_points_bary, sigma=0.02, seed=100+i)
            refined = refine_local(perturbed, time_limit=3.0)
            score = -hard_min_obj(refined.flatten())
            if score > best_score:
                best_score = score
                best_points_bary = refined.copy()

    if best_points_bary is None:
        # Fallback
        best_points_bary = generate_random_init(seed=0)

    # Convert to Cartesian for output
    best_points_cart = bary_to_cart(best_points_bary)
    
    return best_points_cart
