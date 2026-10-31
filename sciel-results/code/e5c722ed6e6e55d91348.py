"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """
    Clamp barycentric coordinates to the valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to valid interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """
    Convert Cartesian coords to barycentric (u, v).
    """
    w = cart[:, 1] / H
    v = cart[:, 0] - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.5, 30):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s * 0.866  # sqrt(3)/2
                if u + v <= 1 - 1e-6 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            bary = clamp_bary(bary)
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

def _soft_min_with_variance(bary_flat, beta=200.0, lam=0.05):
    """
    Hybrid objective: soft-min of areas minus a small variance penalty on smallest areas.
    Encourages balancing the critical triangles.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        sm = np.min(areas)
    else:
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        sm = -log_sum / beta
    
    # Variance on bottom 20%
    n_small = max(1, int(0.2 * len(areas)))
    sorted_areas = np.sort(areas)
    small_areas = sorted_areas[:n_small]
    var_term = np.var(small_areas)
    
    return -(sm - lam * var_term)

def _coordinate_descent_polish(bary, deadline, max_rounds=500):
    """
    Coordinate descent: optimize one point at a time while others are fixed.
    For each point, the objective is the min over all triangles involving that point.
    We use a soft-min over those triangles for smoothness.
    """
    bary = bary.copy()
    current_best = -hard_min_obj(bary.flatten())
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            # Find all triangles involving point i
            # For each triangle (i, j, k), the area depends on bary[i]
            # We optimize bary[i] (2 vars) to maximize the min area over triangles containing i
            # But also need to consider triangles NOT containing i are fixed.
            
            # Get all triangle indices involving point i
            tri_with_i = []
            for t_idx in range(len(tri_indices)):
                if i in tri_indices[t_idx]:
                    tri_with_i.append(t_idx)
            
            if len(tri_with_i) == 0:
                continue
            
            # Get the other two points for each triangle
            others = []
            for t_idx in tri_with_i:
                a, b, c = tri_indices[t_idx]
                if a == i:
                    others.append((b, c))
                elif b == i:
                    others.append((a, c))
                else:
                    others.append((a, b))
            
            # Also get triangles NOT involving i (their areas are fixed)
            tri_without_i = []
            for t_idx in range(len(tri_indices)):
                if i not in tri_indices[t_idx]:
                    tri_without_i.append(t_idx)
            
            fixed_areas = calc_areas_bary(bary)[tri_without_i] if len(tri_without_i) > 0 else np.array([1.0])
            min_fixed = np.min(fixed_areas)
            
            # Define objective: maximize min(areas_of_triangles_with_i, min_fixed)
            # Equivalent to: maximize min(areas_with_i) but capped by min_fixed
            # Use soft min for smoothness
            def obj_i(x):
                bary_new = bary.copy()
                bary_new[i] = x
                bary_new = clamp_bary(bary_new)
                areas_all = calc_areas_bary(bary_new)
                areas_with_i = areas_all[tri_with_i]
                
                # The effective min is min(min(areas_with_i), min_fixed)
                # Use soft min over the combined set
                combined = np.concatenate([areas_with_i, np.array([min_fixed])])
                beta_local = 200.0
                scaled = -beta_local * combined
                max_s = np.max(scaled)
                if max_s > 700:
                    sm = np.min(combined)
                else:
                    log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
                    sm = -log_sum / beta_local
                return -sm
            
            # Optimize point i
            try:
                res = minimize(
                    obj_i,
                    bary[i],
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                bary_new = bary.copy()
                bary_new[i] = res.x
                bary_new = clamp_bary(bary_new)
                
                new_val = -hard_min_obj(bary_new.flatten())
                if new_val > current_best - 1e-12:
                    bary = bary_new
                    current_best = new_val
                    improved = True
            except Exception:
                pass
        
        if not improved and rnd > 20:
            break
    
    return bary, current_best

def _local_polish_bary(points_cart, deadline):
    """
    Local refinement in barycentric coordinates using multiple methods.
    """
    bary = cart_to_bary(points_cart)
    bary = clamp_bary(bary)
    
    current_min_area = -hard_min_obj(bary.flatten())
    
    # Nelder-Mead polish
    if time.time() < deadline - 8:
        try:
            nm_result = minimize(
                hard_min_obj,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            new_bary = clamp_bary(nm_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 5:
        try:
            bary, current_min_area = _coordinate_descent_polish(bary, deadline, max_rounds=200)
        except Exception:
            pass
    
    # L-BFGS-B with high beta soft min for fine tuning
    if time.time() < deadline - 3:
        try:
            lbfgs_result = minimize(
                lambda x: soft_min_obj(x, beta=500.0),
                bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            new_bary = clamp_bary(lbfgs_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    # Final Nelder-Mead with very tight tolerances
    if time.time() < deadline - 2:
        try:
            nm_result2 = minimize(
                hard_min_obj,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            new_bary = clamp_bary(nm_result2.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    cart = bary_to_cart(bary)
    return cart, current_min_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def try_optimize(bary_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        """
        Two-stage optimization:
        1. Smooth landscape (low beta) to find a good basin.
        2. Sharp landscape (high beta) to refine the minimum.
        """
        if time.time() > deadline - 12:
            return
        
        try:
            bounds = [(0, 1)] * (N * 2)
            
            # Stage 1: Exploration with smooth objective
            result1 = minimize(
                lambda x: soft_min_obj(x, beta=beta1),
                bary_init.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 6:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Exploitation with sharper objective
            result2 = minimize(
                lambda x: soft_min_obj(x, beta=beta2),
                result1.x,
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result2.x.reshape(N, 2))
                
        except Exception:
            pass

    def try_variance_optimize(bary_init, maxiter=300):
        """Optimize using the variance-aware smooth objective."""
        if time.time() > deadline - 12:
            return
        
        try:
            bounds = [(0, 1)] * (N * 2)
            
            result = minimize(
                lambda x: _soft_min_with_variance(x, beta=200.0, lam=0.03),
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            
            update_best(result.x.reshape(N, 2))
            
        except Exception:
            pass

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400)
    
    # 2. Try multiple random initializations with different seeds
    num_random_starts = 10
    for i in range(num_random_starts):
        if time.time() > deadline - 35:
            break
        rng_seed = 100 + i * 7
        rand_init = generate_random_init(seed=rng_seed)
        try_optimize(rand_init, method='L-BFGS-B', beta1=40.0, beta2=150.0, maxiter1=150, maxiter2=250)
        
    # 3. Perturb the best found and re-optimize with multiple noise scales
    if best_bary is not None:
        for i in range(8):
            if time.time() > deadline - 20:
                break
            noise = 0.01 + 0.01 * i
            perturbed = generate_perturbed_init(best_bary, noise_scale=noise, seed=200 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300)
    
    # 4. Variance-aware optimization from best
    if best_bary is not None and time.time() < deadline - 18:
        try_variance_optimize(best_bary, maxiter=400)
    
    # 5. Another round of perturbations from the new best
    if best_bary is not None and time.time() < deadline - 15:
        for i in range(4):
            if time.time() > deadline - 12:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.005, seed=300 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta1=150.0, beta2=400.0, maxiter1=150, maxiter2=200)
    
    # 6. Final polish: Nelder-Mead on hard min
    if best_bary is not None and time.time() < deadline - 12:
        try:
            nm_result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            update_best(nm_result.x.reshape(N, 2))
        except Exception:
            pass
    
    # 7. Convert best barycentric to Cartesian and do final local polish
    if best_bary is not None:
        best_cart = bary_to_cart(best_bary)
        # Final local polish in Cartesian space
        if time.time() < deadline - 5:
            best_cart, _ = _local_polish_bary(best_cart, deadline)
        
        # Verify the result is valid
        final_bary = cart_to_bary(best_cart)
        final_bary = clamp_bary(final_bary)
        final_val = -hard_min_obj(final_bary.flatten())
        if final_val > best_val:
            best_val = final_val
            best_bary = final_bary
            best_cart = bary_to_cart(best_bary)
        
        return best_cart
    else:
        # Fallback: return hexagonal init converted to Cartesian
        return bary_to_cart(hex_init)
