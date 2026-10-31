import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

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
    S = 1 - y / H
    u = 0.5 + 0.5 * S - x
    v = S - u
    return np.column_stack([u, v])

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

def calc_min_area(bary):
    """Calculate minimum area efficiently."""
    return np.min(calc_areas_bary(bary))

def clamp_bary(bary, eps=1e-10):
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

def generate_symmetric_inits():
    """
    Generate diverse initial configurations with various symmetric patterns.
    """
    inits = []
    rng = np.random.default_rng(42)
    cx, cy = 0.5, np.sqrt(3)/6
    
    # Pattern A: 1 center + 3 rings of 3 points (total 10) + 1 offset
    for trial in range(15):
        bary = np.zeros((N, 2))
        bary[0] = [1/3, 1/3]
        
        r1 = 0.12 + rng.uniform(-0.03, 0.03)
        r2 = 0.28 + rng.uniform(-0.03, 0.03)
        r3 = 0.42 + rng.uniform(-0.03, 0.03)
        
        for ring, r in enumerate([r1, r2, r3]):
            angle_offset = rng.uniform(0, 2*np.pi/3)
            for i in range(3):
                angle = 2*np.pi*i/3 + angle_offset
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle) * (2/np.sqrt(3))
                bary_pt = cart_to_bary(np.array([[x, y]]))[0]
                bary[1 + ring*3 + i] = bary_pt
        
        bary[10] = [1/3 + rng.uniform(-0.08, 0.08), 1/3 + rng.uniform(-0.08, 0.08)]
        bary = clamp_bary(bary)
        if np.all(bary[:, 0] > 0) and np.all(bary[:, 1] > 0) and np.all(bary[:, 0] + bary[:, 1] < 1):
            inits.append(bary)
    
    # Pattern B: Vertices + edge midpoints + interior
    for trial in range(10):
        bary = np.zeros((N, 2))
        # Near vertices
        bary[0] = [0.05 + rng.uniform(0, 0.05), 0.05 + rng.uniform(0, 0.05)]
        bary[1] = [0.95 - rng.uniform(0, 0.05), 0.05 + rng.uniform(0, 0.05)]
        bary[2] = [0.5, 0.9 - rng.uniform(0, 0.1)]
        
        # Edge midpoints
        bary[3] = [0.25 + rng.uniform(-0.03, 0.03), 0.05 + rng.uniform(-0.03, 0.03)]
        bary[4] = [0.75 + rng.uniform(-0.03, 0.03), 0.05 + rng.uniform(-0.03, 0.03)]
        bary[5] = [0.25 + rng.uniform(-0.03, 0.03), 0.45 + rng.uniform(-0.03, 0.03)]
        
        # Interior points
        bary[6] = [1/3 + rng.uniform(-0.05, 0.05), 1/3 + rng.uniform(-0.05, 0.05)]
        bary[7] = [0.35 + rng.uniform(-0.05, 0.05), 0.25 + rng.uniform(-0.05, 0.05)]
        bary[8] = [0.65 + rng.uniform(-0.05, 0.05), 0.25 + rng.uniform(-0.05, 0.05)]
        bary[9] = [0.5 + rng.uniform(-0.05, 0.05), 0.35 + rng.uniform(-0.05, 0.05)]
        bary[10] = [0.45 + rng.uniform(-0.05, 0.05), 0.15 + rng.uniform(-0.05, 0.05)]
        
        bary = clamp_bary(bary)
        if np.all(bary[:, 0] > 0) and np.all(bary[:, 1] > 0) and np.all(bary[:, 0] + bary[:, 1] < 1):
            inits.append(bary)
    
    # Pattern C: Random but with minimum pairwise distance enforcement
    for trial in range(15):
        bary = np.zeros((N, 2))
        bary[0] = [1/3, 1/3]
        
        for i in range(1, N):
            attempts = 0
            while attempts < 100:
                u = rng.random()
                v = rng.random() * (1 - u)
                candidate = np.array([u, v])
                
                # Check minimum distance to existing points
                min_dist = np.inf
                for j in range(i):
                    d = np.linalg.norm(candidate - bary[j])
                    min_dist = min(min_dist, d)
                
                if min_dist > 0.15 or attempts > 50:
                    bary[i] = candidate
                    break
                attempts += 1
        
        bary = clamp_bary(bary)
        if np.all(bary[:, 0] > 0) and np.all(bary[:, 1] > 0) and np.all(bary[:, 0] + bary[:, 1] < 1):
            inits.append(bary)
    
    # Add fully random starts
    for i in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    curr_val = calc_min_area(bary_current)
    
    # Stage 0: Pre-optimization with pairwise repulsion to spread points
    if time.time() - start_time < time_limit * 0.15:
        def repulsion_obj(bary_flat):
            bary = bary_flat.reshape(N, 2)
            bary = clamp_bary(bary)
            areas = calc_areas_bary(bary)
            min_a = np.min(areas)
            
            # Add pairwise distance penalty
            cart = bary_to_cart(bary)
            pair_dists = []
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(cart[i] - cart[j])
                    pair_dists.append(d)
            pair_dists = np.array(pair_dists)
            min_pair_dist = np.min(pair_dists)
            
            return -min_a - 0.1 * min_pair_dist
        
        try:
            result0 = minimize(
                repulsion_obj,
                bary_current.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            new_bary = clamp_bary(result0.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
    
    # Stage 1: Smooth soft-min minimization with increasing beta
    for beta in [20.0, 100.0, 500.0, 2000.0]:
        if time.time() - start_time > time_limit * 0.45:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_bary = clamp_bary(result1.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
        
    # Stage 2: Hard min refinement with SLSQP
    if time.time() - start_time < time_limit * 0.75:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 8000, 'ftol': 1e-18}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
            
    # Stage 3: Active-set focused optimization (multiple passes)
    for pass_num in range(3):
        if time.time() - start_time > time_limit * 0.9:
            break
        try:
            areas = calc_areas_bary(bary_current)
            min_area = np.min(areas)
            threshold = min_area * 1.03
            critical_idx = np.where(areas <= threshold)[0]
            
            if len(critical_idx) > 0:
                def active_set_obj(bary_flat):
                    bary = bary_flat.reshape(N, 2)
                    bary = clamp_bary(bary)
                    all_areas = calc_areas_bary(bary)
                    min_a = np.min(all_areas)
                    crit_areas = all_areas[critical_idx]
                    # Use a smoother penalty
                    dev_penalty = np.sum(np.maximum(0, target - crit_areas)**2)
                    return -min_a + 0.05 * dev_penalty
                
                target = min_area * 1.01
                result3 = minimize(
                    active_set_obj,
                    bary_current.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 5000, 'ftol': 1e-16}
                )
                new_bary = clamp_bary(result3.x.reshape(N, 2))
                new_val = calc_min_area(new_bary)
                if new_val > curr_val:
                    bary_current = new_bary
                    curr_val = new_val
        except Exception:
            pass
            
    # Stage 4: Nelder-Mead for final polish
    if time.time() - start_time < time_limit * 0.95:
        try:
            result4 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            new_bary = clamp_bary(result4.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
        
    # Stage 5: Adaptive local search with decreasing step size
    if time.time() - start_time < time_limit:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        step_size = 0.03
        for iteration in range(150):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for _ in range(15):
                perturbation = rng.normal(0, step_size, (N, 2))
                new_bary = clamp_bary(bary_current + perturbation)
                new_val = calc_min_area(new_bary)
                if new_val > curr_val:
                    bary_current = new_bary
                    curr_val = new_val
                    improved = True
                    break
            if not improved:
                step_size *= 0.65
            if step_size < 1e-7:
                break
                
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = calc_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate diverse initial points
    rng = np.random.default_rng(123)
    inits = []
    
    # 1. Symmetric starts
    sym_inits = generate_symmetric_inits()
    inits.extend(sym_inits)
    
    # 2. Random starts
    for i in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # Deduplicate inits (approximately)
    unique_inits = []
    for init in inits:
        init_clamped = clamp_bary(init)
        is_unique = True
        for existing in unique_inits:
            if np.allclose(init_clamped, existing, atol=0.02):
                is_unique = False
                break
        if is_unique:
            unique_inits.append(init_clamped)
    
    inits = unique_inits
    
    # Sort inits by initial quality (best first)
    inits_with_vals = [(init, calc_min_area(init)) for init in inits]
    inits_with_vals.sort(key=lambda x: -x[1])
    inits = [x[0] for x in inits_with_vals]
    
    # Refine each initial point with time budget
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Allocate time based on rank and remaining time
        remaining_inits = len(inits) - i
        if i < 3:
            base_time = 7.0
        elif i < 8:
            base_time = 5.0
        else:
            base_time = 3.0
            
        time_per_init = min(base_time, (deadline - time.time() - 15) / max(1, remaining_inits))
        if time_per_init < 0.8:
            time_per_init = 0.8
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # If we have time, do a global search with differential evolution
    if best_bary is not None and time.time() < deadline - 20:
        try:
            def de_objective(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            bounds_de = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=80,
                tol=1e-10,
                popsize=30,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                update_interval=1,
                callback=None,
                init='sobol'
            )
            if result_de.success or result_de.fun < hard_min_obj(best_bary.flatten()):
                bary_de = clamp_bary(result_de.x.reshape(N, 2))
                if update_best(bary_de):
                    if time.time() < deadline - 8:
                        refined_de = refine_local(bary_de, time_limit=min(8.0, deadline - time.time() - 4))
                        update_best(refined_de)
        except Exception:
            pass
        
    # Final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 10:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-18}
            )
            if result_slsqp.fun < hard_min_obj(best_bary.flatten()):
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 5:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 25000, 'xatol': 1e-15, 'fatol': 1e-20, 'adaptive': True}
                )
                if result_nm.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
            
        # Final adaptive perturbation search
        if best_bary is not None and time.time() < deadline - 1:
            rng_final = np.random.default_rng(999)
            step = 0.02
            for _ in range(50):
                if time.time() > deadline - 1:
                    break
                improved = False
                for _ in range(8):
                    pert = rng_final.normal(0, step, (N, 2))
                    new_bary = clamp_bary(best_bary + pert)
                    if update_best(new_bary):
                        improved = True
                        break
                if not improved:
                    step *= 0.75
                if step < 1e-8:
                    break

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
