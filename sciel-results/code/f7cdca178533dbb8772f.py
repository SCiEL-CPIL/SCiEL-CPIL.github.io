import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

def _disk_packing_init(seed=7):
    """
    Generate an initial configuration by placing N points as far apart as possible
    (maximize minimum pairwise distance) inside the triangle, then use that as a
    starting point for the triangle-area optimization. The intuition is that well-
    separated points tend to form larger minimum-area triangles.
    """
    rng = np.random.default_rng(seed)
    
    # Try multiple random starts and keep the best
    best_bary = None
    best_min_dist = -np.inf
    
    for attempt in range(5):
        # Start with random points
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        
        # Iteratively push points apart (simple repulsion) with annealing
        for it in range(300):
            bary = clamp_bary(bary)
            # Compute pairwise distances in Cartesian
            cart = bary_to_cart(bary)
            dists = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Force on each point: sum of unit vectors away from neighbors, weighted by 1/d^2
            forces = np.zeros_like(cart)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = dists[i, j]
                    if d < 1e-6:
                        continue
                    forces[i] += (cart[i] - cart[j]) / (d * d)
            
            # Annealing: start with larger steps, decrease over time
            step_size = 0.02 * (1.0 - it / 300.0) + 0.002
            bary += step_size * forces
            bary = clamp_bary(bary)
        
        # Evaluate
        cart = bary_to_cart(bary)
        dists = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_bary = bary.copy()
    
    return best_bary

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

def epigraph_obj(params):
    """
    Epigraph formulation: minimize t subject to areas >= t.
    params = [bary (N*2), t]
    """
    bary_flat = params[:N*2]
    t = params[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas < t
    violation = np.maximum(0, t - areas)
    penalty = np.sum(violation**2) * 1000
    return -t + penalty

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.35, 0.4, 0.45]:
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
            for _ in range(10):
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

def generate_symmetric_init():
    """
    Generate a symmetric initial configuration based on known Heilbronn structures.
    """
    # Multiple symmetric patterns to try
    candidates = []
    
    # Pattern 1: 3 corners + 3 edge midpoints + 5 interior
    bary1 = np.array([
        [0.02, 0.02],    # corner 1
        [0.02, 0.98],    # corner 2  
        [0.98, 0.02],    # corner 3
        [0.02, 0.5],     # edge
        [0.5, 0.02],     # edge
        [0.49, 0.49],    # center
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.5]
    ])
    candidates.append(clamp_bary(bary1))
    
    # Pattern 2: More spread out
    bary2 = np.array([
        [0.03, 0.03],
        [0.03, 0.97],
        [0.97, 0.03],
        [0.03, 0.45],
        [0.45, 0.03],
        [0.55, 0.45],
        [0.2, 0.25],
        [0.8, 0.2],
        [0.5, 0.55],
        [0.2, 0.7],
        [0.7, 0.3]
    ])
    candidates.append(clamp_bary(bary2))
    
    # Pattern 3: Even more spread
    bary3 = np.array([
        [0.05, 0.05],
        [0.05, 0.95],
        [0.95, 0.05],
        [0.05, 0.4],
        [0.4, 0.05],
        [0.6, 0.4],
        [0.2, 0.2],
        [0.8, 0.2],
        [0.5, 0.5],
        [0.2, 0.65],
        [0.65, 0.25]
    ])
    candidates.append(clamp_bary(bary3))
    
    # Pick the best candidate
    best_bary = candidates[0]
    best_val = -hard_min_obj(best_bary.flatten())
    for bary in candidates[1:]:
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary
    
    return best_bary

def _voronoi_area_var_penalty(bary, weight=1.0):
    """
    Compute a penalty based on the variance of approximate Voronoi cell areas.
    Uses the fact that in a triangle, the Voronoi cell area for point i is
    approximately proportional to the area of the triangle formed by point i
    and its two nearest neighbors. We use a simpler proxy: the area of the
    smallest triangle containing point i as a vertex.
    
    Actually, a better approach: compute the "influence area" for each point
    as the area of the convex hull cell. For simplicity, we use the sum of
    triangle areas where point i is the "apex" (opposite the base formed by
    two other points) weighted by how central point i is to that triangle.
    
    Simplest effective proxy: the area of the triangle formed by point i and
    its two closest neighbors. This approximates the Voronoi cell area.
    """
    cart = bary_to_cart(bary)
    dists = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # For each point, find its 2 nearest neighbors
    nn_idx = np.argsort(dists, axis=1)[:, :2]
    
    # Compute area of triangle (i, nn1, nn2) for each i
    cell_areas = np.zeros(N)
    for i in range(N):
        j, k = nn_idx[i]
        # Area of triangle in barycentric coords
        ui = bary[i, 0]; vi = bary[i, 1]; wi = 1 - ui - vi
        uj = bary[j, 0]; vj = bary[j, 1]; wj = 1 - uj - vj
        uk = bary[k, 0]; vk = bary[k, 1]; wk = 1 - uk - vk
        det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
        cell_areas[i] = abs(det)
    
    # Penalize variance of cell areas (encourage uniform distribution)
    if np.min(cell_areas) < 1e-15:
        return weight * 1.0
    var = np.var(cell_areas) / (np.mean(cell_areas) ** 2 + 1e-15)
    return weight * var


def separation_aware_obj(bary_flat, gamma=10.0, voronoi_weight=0.02):
    """
    Combined objective: maximize min triangle area while also encouraging
    good pairwise separation and uniform Voronoi cell distribution.
    Uses a smooth surrogate for both terms.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Smooth min of areas (log-sum-exp)
    if np.any(areas < 1e-15):
        return 1e10
    scaled_a = -gamma * areas
    max_sa = np.max(scaled_a)
    if max_sa > 700:
        soft_min_area = -np.min(areas)
    else:
        log_sum_a = max_sa + np.log(np.sum(np.exp(scaled_a - max_sa)))
        soft_min_area = -log_sum_a / gamma
    
    # Pairwise distances in Cartesian
    cart = bary_to_cart(bary)
    dists = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Smooth min of distances
    scaled_d = -gamma * dists
    max_sd = np.max(scaled_d)
    if max_sd > 700:
        soft_min_dist = -min_dist
    else:
        log_sum_d = max_sd + np.log(np.sum(np.exp(scaled_d - max_sd)))
        soft_min_dist = -log_sum_d / gamma
    
    # Voronoi area variance penalty
    voronoi_pen = _voronoi_area_var_penalty(bary, voronoi_weight)
    
    # Combine: primary is area, secondary is separation, tertiary is uniformity
    return -(soft_min_area + 0.05 * soft_min_dist) - voronoi_pen

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

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    # Stage 1: Graduated soft-min minimization (multiple passes with increasing beta)
    for step_idx, beta in enumerate(beta_schedule):
        if time.time() - start_time > time_limit * 0.40:
            break
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            # Only accept if it improves the hard min value
            old_val = -hard_min_obj(bary_current.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > old_val:
                bary_current = new_bary
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.40:
        return bary_current
    
    # Stage 2: Separation-aware + Voronoi refinement to escape local minima
    try:
        result_sep = minimize(
            separation_aware_obj,
            bary_current.flatten(),
            args=(10.0, 0.02),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        new_bary = clamp_bary(result_sep.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.55:
        return bary_current
    
    # Stage 2b: Second separation-aware pass with different parameters
    if time.time() - start_time < time_limit * 0.60:
        try:
            result_sep2 = minimize(
                separation_aware_obj,
                bary_current.flatten(),
                args=(20.0, 0.01),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_bary = clamp_bary(result_sep2.x.reshape(N, 2))
            old_val = -hard_min_obj(bary_current.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > old_val:
                bary_current = new_bary
        except Exception:
            pass
        
    # Stage 3: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-17, 'adaptive': True}
        )
        new_bary = clamp_bary(result2.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.80:
        return bary_current
    
    # Stage 4: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-15}
        )
        new_bary = clamp_bary(result3.x.reshape(N, 2))
        old_val = -hard_min_obj(bary_current.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            bary_current = new_bary
    except Exception:
        pass
        
    return bary_current

def refine_epigraph(bary_init, time_limit=5.0):
    """
    Refine using epigraph formulation with auxiliary variable t.
    """
    start_time = time.time()
    
    # Initial guess for t
    areas = calc_areas_bary(clamp_bary(bary_init))
    t_init = np.min(areas) * 0.5
    
    # Full params: bary (22) + t (1) = 23
    bounds_full = [(0, 1)] * (N * 2) + [(0, 1)]
    
    params_init = np.concatenate([clamp_bary(bary_init).flatten(), [t_init]])
    
    try:
        result = minimize(
            epigraph_obj,
            params_init,
            method='SLSQP',
            bounds=bounds_full,
            options={'maxiter': 800, 'ftol': 1e-15}
        )
        new_bary = clamp_bary(result.x[:N*2].reshape(N, 2))
        old_val = -hard_min_obj(bary_init.flatten())
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > old_val:
            return new_bary
    except Exception:
        pass
    
    # Try with a better t initialization
    if time.time() - start_time < time_limit * 0.5:
        t_init2 = np.min(areas) * 0.8
        params_init2 = np.concatenate([clamp_bary(bary_init).flatten(), [t_init2]])
        try:
            result2 = minimize(
                epigraph_obj,
                params_init2,
                method='SLSQP',
                bounds=bounds_full,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            new_bary2 = clamp_bary(result2.x[:N*2].reshape(N, 2))
            old_val = -hard_min_obj(bary_init.flatten())
            new_val = -hard_min_obj(new_bary2.flatten())
            if new_val > old_val:
                return new_bary2
        except Exception:
            pass
    
    return bary_init

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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=7.0)
    update_best(refined_hex)
    
    # 2. Try symmetric init
    sym_init = generate_symmetric_init()
    refined_sym = refine_local(sym_init, time_limit=7.0)
    update_best(refined_sym)
    
    # 3. Try disk-packing-inspired init (maximize separation first)
    disk_init = _disk_packing_init(seed=42)
    refined_disk = refine_local(disk_init, time_limit=7.0)
    update_best(refined_disk)
    
    # 4. Epigraph refinement from best so far
    if best_bary is not None and time.time() < deadline - 60:
        refined_epi = refine_epigraph(best_bary, time_limit=6.0)
        update_best(refined_epi)
    
    # 5. Global Search with Differential Evolution
    if time.time() < deadline - 45:
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
                maxiter=40,
                popsize=15,
                tol=1e-7,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=7.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 6. Multi-start: random + perturbed restarts with local refinement
    restart_count = 0
    while time.time() < deadline - 18:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution with varying sigma
            sigma = 0.03 + 0.02 * (restart_count % 3)
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            # Moderate perturbation
            init = generate_perturbed_init(best_bary, sigma=0.06, seed=4000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        # Use shorter time limit for more restarts
        tl = 2.5 if restart_count < 10 else 2.0
        refined = refine_local(init, time_limit=tl)
        update_best(refined)
        
        # Also try epigraph refinement on the best
        if best_bary is not None and time.time() < deadline - 15:
            refined_epi = refine_epigraph(best_bary, time_limit=2.5)
            update_best(refined_epi)

    # 7. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 6:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-16}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 1.0:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-17}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass
        
        # Final Nelder-Mead micro-polish
        if best_bary is not None and time.time() < deadline - 0.3:
            try:
                result_final_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                update_best(clamp_bary(result_final_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
