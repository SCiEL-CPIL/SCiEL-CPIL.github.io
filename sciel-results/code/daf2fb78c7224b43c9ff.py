import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

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
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    
    # Cross-product formulation: det = (uj-ui)*(vk-vi) - (vj-vi)*(uk-ui)
    # This is faster as it avoids computing w explicitly
    du = uj - ui
    dv = vj - vi
    du2 = uk - ui
    dv2 = vk - vi
    
    det = du * dv2 - dv * du2
    
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

def epigraph_obj(x_flat):
    """
    Epigraph formulation: minimize -t subject to t <= area_i for all i.
    x_flat = [bary_0_0, bary_0_1, ..., bary_10_0, bary_10_1, t]
    """
    bary = x_flat[:N*2].reshape(N, 2)
    t = x_flat[N*2]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for violations: t > area_i
    violations = np.maximum(t - areas, 0)
    penalty = 1e6 * np.sum(violations**2)
    
    return -t + penalty

def soft_min_obj_grad(bary_flat, beta=50.0):
    """
    Smooth surrogate with analytical gradient computation.
    Returns (value, gradient) for use with L-BFGS-B.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        grad = np.zeros_like(bary_flat)
        return 1e10, grad
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        # Fall back to hard min with finite differences
        val = -np.min(areas)
        grad = np.zeros_like(bary_flat)
        return val, grad
        
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    log_sum = max_scaled + np.log(sum_exp)
    soft_min = -log_sum / beta
    val = -soft_min
    
    # Gradient via chain rule
    # d(soft_min)/d(area_i) = -exp_scaled[i] / (beta * sum_exp)
    # d(area_i)/d(bary) needs Jacobian
    weights = exp_scaled / sum_exp  # softmax weights
    
    # Compute gradient of each area w.r.t. barycentric coords
    # area_i = |det| where det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    # For simplicity, use finite differences for gradient
    grad = np.zeros_like(bary_flat)
    eps_fd = 1e-7
    for i in range(N * 2):
        bary_pert = bary_flat.copy()
        bary_pert[i] += eps_fd
        bary_pert_clamped = clamp_bary(bary_pert.reshape(N, 2)).flatten()
        areas_pert = calc_areas_bary(bary_pert_clamped.reshape(N, 2))
        soft_min_pert = -(-np.max(-beta * areas_pert) + np.log(np.sum(np.exp(-beta * areas_pert - np.max(-beta * areas_pert))))) / beta
        grad[i] = (soft_min_pert - soft_min) / eps_fd
    
    return val, -grad

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration with multiple scales.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.18, 0.22, 0.25, 0.28, 0.32, 0.36, 0.40, 0.45]:
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
            # Try multiple random subsets
            rng = np.random.default_rng(42 + int(scale * 100))
            for trial in range(8):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    # Also try a structured grid
    for n_rows in [3, 4, 5]:
        pts = []
        for i in range(n_rows + 1):
            w_val = i / n_rows
            num_cols = int((1 - w_val) * n_rows) + 1
            for j in range(num_cols):
                u_val = j / (n_rows if num_cols > 1 else 1) * (1 - w_val)
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            rng = np.random.default_rng(100 + n_rows)
            for trial in range(5):
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

def coordinated_ascent(bary_init, time_limit=2.0):
    """
    Coordinated ascent: iteratively move each point to maximize the minimum area
    while keeping other points fixed. Targets active constraints directly.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    best_val = -hard_min_obj(bary_current.flatten())
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                    break
            
            # Optimize point i while keeping others fixed
            def obj_i(x_i):
                bary_temp = bary_current.copy()
                bary_temp[i] = x_i
                bary_temp = clamp_bary(bary_temp)
                areas = calc_areas_bary(bary_temp)
                return -np.min(areas)
            
            bounds_i = [(0, 1), (0, 1)]
            try:
                result_i = minimize(
                    obj_i,
                    bary_current[i],
                    method='Nelder-Mead',
                    bounds=bounds_i,
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                new_bary = bary_current.copy()
                new_bary[i] = result_i.x
                new_bary = clamp_bary(new_bary)
                new_val = -hard_min_obj(new_bary.flatten())
                
                if new_val > best_val:
                    bary_current = new_bary
                    best_val = new_val
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return bary_current

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with adaptive time allocation.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    
    def elapsed():
        return time.time() - start_time
    
    # Stage 1: Graduated soft-min minimization (beta = 15 -> 40 -> 120)
    for beta in [15.0, 40.0, 120.0]:
        if elapsed() > time_limit * 0.2:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            new_val = -hard_min_obj(clamp_bary(result1.x.reshape(N, 2)).flatten())
            if new_val > current_val:
                bary_current = clamp_bary(result1.x.reshape(N, 2))
                current_val = new_val
        except Exception:
            pass
    
    if elapsed() > time_limit * 0.3:
        return bary_current
        
    # Stage 2: Epigraph formulation with auxiliary t variable
    if elapsed() < time_limit * 0.45:
        try:
            areas_init = calc_areas_bary(bary_current)
            t_init = np.min(areas_init) * 0.98
            x_init = np.concatenate([bary_current.flatten(), [t_init]])
            
            bounds_epigraph = [(0, 1)] * (N * 2) + [(0, 1)]
            
            result_ep = minimize(
                epigraph_obj,
                x_init,
                method='L-BFGS-B',
                bounds=bounds_epigraph,
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            
            new_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass
    
    if elapsed() > time_limit * 0.5:
        return bary_current
    
    # Stage 3: Coordinated ascent to target active constraints
    if elapsed() < time_limit * 0.7:
        try:
            bary_new = coordinated_ascent(bary_current, time_limit=time_limit * 0.12)
            new_val = -hard_min_obj(bary_new.flatten())
            if new_val > current_val:
                bary_current = bary_new
                current_val = new_val
        except Exception:
            pass
    
    if elapsed() > time_limit * 0.75:
        return bary_current
    
    # Stage 4: Hard min refinement with Nelder-Mead
    if elapsed() < time_limit * 0.9:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass
        
    if elapsed() > time_limit * 0.92:
        return bary_current
    
    # Stage 5: SLSQP polish with tighter tolerance
    if elapsed() < time_limit * 0.98:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            new_bary = clamp_bary(result3.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
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
    
    def remaining():
        return deadline - time.time()

    # 1. Try multiple hexagonal inits with different scales
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=6.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution (reduced iterations for speed)
    if remaining() > 30:
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
                maxiter=25,
                popsize=10,
                tol=1e-5,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=6.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while remaining() > 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution with varying sigma
            sigma = 0.03 + 0.02 * (restart_count % 5)
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count)
        elif restart_count % 3 == 1:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        else:
            # Structured perturbation
            init = generate_hexagonal_init()
            init = generate_perturbed_init(init, sigma=0.05, seed=4000 + restart_count)
        
        # Adaptive time limit based on remaining budget
        tl = min(3.0, remaining() * 0.25)
        if tl < 1.0:
            break
        refined = refine_local(init, time_limit=tl)
        update_best(refined)

    # 4. Final high-precision polish
    if best_bary is not None and remaining() > 10:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-14}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and remaining() > 6:
            try:
                best_bary = coordinated_ascent(best_bary, time_limit=min(3.0, remaining() * 0.4))
                update_best(best_bary)
            except Exception:
                pass
        
        if best_bary is not None and remaining() > 3:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and remaining() > 1.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and remaining() > 0.5:
            try:
                areas_final = calc_areas_bary(best_bary)
                t_final = np.min(areas_final)
                x_final = np.concatenate([best_bary.flatten(), [t_final]])
                bounds_ep = [(0, 1)] * (N * 2) + [(0, 1)]
                result_ep_final = minimize(
                    epigraph_obj,
                    x_final,
                    method='L-BFGS-B',
                    bounds=bounds_ep,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                new_bary = clamp_bary(result_ep_final.x[:N*2].reshape(N, 2))
                update_best(new_bary)
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
