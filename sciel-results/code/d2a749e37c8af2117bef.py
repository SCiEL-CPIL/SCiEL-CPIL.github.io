import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay
from itertools import combinations
import time

N = 11
TIME_LIMIT = 140.0  # Safety margin for 150s limit

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

def epigraph_obj(bary_flat, t_init=0.01):
    """
    Epigraph formulation: minimize -t subject to t <= all areas.
    We penalize violations where t > area_i.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # We want to maximize t such that t <= min(areas)
    # So we minimize -(t) + penalty * sum(max(0, t - area_i)^2)
    # But we need t as a variable... 
    # Alternative: just use a very high beta soft min which approximates the epigraph
    # For now, let's stick with the graduated soft-min approach which showed improvement
    return soft_min_obj(bary_flat, beta=200.0)

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

def delaunay_perturb(bary, rng, scale=0.03):
    """
    Perturb points using Delaunay triangulation structure.
    Move points in directions that maintain or improve the Delaunay property.
    """
    cart = bary_to_cart(bary)
    try:
        tri = Delaunay(cart)
        simplices = tri.simplices
    except Exception:
        return generate_perturbed_init(bary, sigma=scale, seed=None)
    
    # For each point, identify its neighbors in the Delaunay triangulation
    # and perturb towards the centroid of its neighbor points
    new_bary = bary.copy()
    for i in range(N):
        # Find neighbors of point i
        neighbors = set()
        for simplex in simplices:
            if i in simplex:
                for j in simplex:
                    if j != i:
                        neighbors.add(j)
        
        if neighbors:
            # Centroid of neighbors in Cartesian
            neighbor_pts = cart[list(neighbors)]
            centroid = neighbor_pts.mean(axis=0)
            
            # Direction from point i to centroid
            direction = centroid - cart[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            
            # Small perturbation towards centroid
            step = rng.uniform(0, scale)
            cart_new = cart[i] + direction * step
            cart_new = np.clip(cart_new, [0, 0], [1, H])
            
            # Convert back to barycentric (approximate)
            # x = v + 0.5*w, y = w*H => w = y/H, v = x - 0.5*w, u = 1 - v - w
            w_val = cart_new[1] / H
            v_val = cart_new[0] - 0.5 * w_val
            u_val = 1.0 - v_val - w_val
            
            if u_val >= 0 and v_val >= 0 and w_val >= 0:
                new_bary[i] = [u_val, v_val]
    
    return clamp_bary(new_bary)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Graduated soft-min minimization (β=20 → 50 → 200)
    for beta in [20.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.9:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=min(8.0, deadline - start_time - 30))
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 35:
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
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=min(8.0, deadline - start_time - 25))
            update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 12:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Delaunay-guided perturbation of best solution
            init = delaunay_perturb(best_bary, rng, scale=0.03)
        elif best_bary is not None and restart_count % 2 == 0:
            # Standard perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            # Random restart
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=min(4.0, deadline - start_time - 10))
        update_best(refined)
    
    # 4. Basin hopping on best found
    if best_bary is not None and time.time() < deadline - 20:
        bh_time = min(25.0, deadline - start_time - 15.0)
        if bh_time > 5:
            try:
                start_bh = time.time()
                rng = np.random.default_rng(123)
                current_bary = best_bary.copy()
                best_bh_bary = best_bary.copy()
                best_bh_val = best_val
                
                temperature = 0.03
                pert_scale = 0.05
                
                while time.time() - start_bh < bh_time:
                    # Alternate between standard and Delaunay-guided perturbation
                    if rng.random() < 0.3:
                        perturbed = delaunay_perturb(current_bary, rng, scale=pert_scale)
                    else:
                        noise = rng.normal(0, pert_scale, current_bary.shape)
                        perturbed = clamp_bary(current_bary + noise)
                    
                    # Local refinement
                    refined = refine_local(perturbed, time_limit=min(3.0, bh_time - (time.time() - start_bh)))
                    refined_val = -hard_min_obj(refined.flatten())
                    
                    # Accept if better or with Metropolis
                    if refined_val > best_bh_val:
                        best_bh_val = refined_val
                        best_bh_bary = refined.copy()
                        current_bary = refined.copy()
                    elif np.log(rng.random()) < (refined_val - best_bh_val) / temperature:
                        current_bary = refined.copy()
                    
                    temperature *= 0.97
                    pert_scale *= 0.99
                    
                    if time.time() - start_bh > bh_time * 0.95:
                        break
                
                update_best(best_bh_bary)
            except Exception:
                pass
    
    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
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
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass if time permits
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
