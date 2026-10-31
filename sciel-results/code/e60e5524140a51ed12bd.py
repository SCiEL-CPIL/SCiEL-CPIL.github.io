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

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = 2.0 * y / H
    u = 1.0 - v - x
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

def get_areas_cartesian(cart):
    """Compute all triangle areas in Cartesian coordinates."""
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    return 0.5 * np.abs(cross)

def calc_min_area_cartesian(points):
    """Calculate the minimum area of all triangles formed by N points."""
    return np.min(get_areas_cartesian(points))

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

def epigraph_obj(bary_t_flat, penalty=100.0):
    """
    Epigraph formulation: optimize (bary, t) where t is the minimum area.
    Maximize t subject to all areas >= t.
    We minimize -t + penalty * sum(max(0, t - area_i)^2)
    This provides a smoother landscape than pure soft-min.
    """
    bary = bary_t_flat[:N*2].reshape(N, 2)
    t = bary_t_flat[N*2]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for constraints area_i < t
    violations = np.maximum(0, t - areas)
    penalty_term = np.sum(violations**2)
    
    # We want to maximize t, so minimize -t
    # But we also penalize violations
    return -t + penalty * penalty_term

def active_triangles(cart, threshold_frac=0.05, max_count=20):
    """
    Find triangles with areas near the minimum (active constraints).
    Returns list of (i, j, k) indices.
    """
    areas = get_areas_cartesian(cart)
    min_area = np.min(areas)
    if min_area <= 0:
        return [(0, 1, 2)]
    
    threshold = min_area * (1 + threshold_frac)
    active = np.where(areas <= threshold)[0]
    
    if len(active) > max_count:
        # Take the smallest ones
        order = np.argsort(areas[active])
        active = active[order[:max_count]]
    
    return [(tri_idx[t, 0], tri_idx[t, 1], tri_idx[t, 2]) for t in active]

def coordinated_ascent(bary_init, time_limit=10.0):
    """
    Coordinated ascent: iteratively optimize individual points to increase
    the minimum area, focusing on active constraints.
    """
    start_time = time.time()
    bary = bary_init.copy()
    
    def current_min():
        cart = bary_to_cart(bary)
        return calc_min_area_cartesian(cart)
    
    best_val = current_min()
    
    max_rounds = 50
    for round_idx in range(max_rounds):
        if time.time() - start_time > time_limit - 2:
            break
        
        cart = bary_to_cart(bary)
        areas = get_areas_cartesian(cart)
        min_area = np.min(areas)
        
        # Get active triangles
        active = active_triangles(cart, threshold_frac=0.03, max_count=15)
        
        # For each point, try to move it to improve the minimum
        improved = False
        for pt_idx in range(N):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Collect constraints involving this point
            constraints = []
            for (i, j, k) in active:
                involved = set([i, j, k])
                if pt_idx in involved:
                    constraints.append((i, j, k))
            
            if not constraints:
                continue
            
            # Try small perturbations in barycentric space
            best_delta = None
            best_area = calc_min_area_cartesian(cart)
            
            # Random directions
            rng = np.random.default_rng(round_idx * 1000 + pt_idx)
            for attempt in range(30):
                # Random direction in 2D barycentric
                angle = rng.uniform(0, 2 * np.pi)
                magnitude = rng.uniform(0.01, 0.05)
                du = magnitude * np.cos(angle)
                dv = magnitude * np.sin(angle)
                
                new_bary = bary.copy()
                new_bary[pt_idx, 0] += du
                new_bary[pt_idx, 1] += dv
                new_bary = clamp_bary(new_bary)
                
                new_cart = bary_to_cart(new_bary)
                new_min = calc_min_area_cartesian(new_cart)
                
                if new_min > best_area + 1e-12:
                    best_area = new_min
                    best_delta = (du, dv)
            
            # Also try gradient-like direction: move away from centroid of active triangles
            if best_delta is None:
                # Compute average position of other points in active triangles
                other_pts = []
                for (i, j, k) in constraints:
                    for idx in [i, j, k]:
                        if idx != pt_idx:
                            other_pts.append(idx)
                
                if other_pts:
                    other_cart = cart[list(set(other_pts))]
                    centroid = np.mean(other_cart, axis=0)
                    my_cart = cart[pt_idx]
                    
                    # Direction away from centroid
                    direction = my_cart - centroid
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction = direction / norm
                        
                        # Convert cartesian direction to barycentric
                        dx_dir = direction[0]
                        dy_dir = direction[1]
                        du_est = -dy_dir / (2 * H) - dx_dir
                        dv_est = dx_dir - dy_dir / (2 * H)
                        
                        for scale in [0.02, 0.01, 0.005]:
                            new_bary = bary.copy()
                            new_bary[pt_idx, 0] += scale * du_est
                            new_bary[pt_idx, 1] += scale * dv_est
                            new_bary = clamp_bary(new_bary)
                            
                            new_cart = bary_to_cart(new_bary)
                            new_min = calc_min_area_cartesian(new_cart)
                            
                            if new_min > best_area + 1e-12:
                                best_area = new_min
                                best_delta = (scale * du_est, scale * dv_est)
            
            if best_delta is not None:
                bary[pt_idx, 0] += best_delta[0]
                bary[pt_idx, 1] += best_delta[1]
                bary = clamp_bary(bary)
                improved = True
        
        if not improved:
            break
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with graduated soft-min.
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

    # Stage 4: Epigraph formulation
    if time.time() - start_time < time_limit * 0.85:
        try:
            # Initialize t with current min area
            cart = bary_to_cart(bary_current)
            t_init = calc_min_area_cartesian(cart)
            
            bary_t_init = np.concatenate([bary_current.flatten(), [t_init]])
            
            # Bounds for epigraph: bary in [0,1], t in [0, 0.2]
            epigraph_bounds = [(0, 1)] * (N * 2) + [(0, 0.2)]
            
            result4 = minimize(
                epigraph_obj,
                bary_t_init,
                args=(100.0,),
                method='L-BFGS-B',
                bounds=epigraph_bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result4.success:
                new_bary = clamp_bary(result4.x[:N*2].reshape(N, 2))
                new_t = result4.x[N*2]
                # Verify the t value is achievable
                cart = bary_to_cart(new_bary)
                actual_min = calc_min_area_cartesian(cart)
                if actual_min > -hard_min_obj(bary_current.flatten()):
                    bary_current = new_bary
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Hard min refinement with Nelder-Mead
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.95:
        return bary_current

    # Stage 6: Final high-precision Nelder-Mead polish
    try:
        result6 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result6.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

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
            # Randomly select N points
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pts), size=N, replace=False)
            bary = np.array([pts[i] for i in indices])
            bary = clamp_bary(bary)
            
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary
    
    if best_bary is None:
        # Fallback to random
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
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

    # 1. Global Search using Differential Evolution
    bounds = [(0, 1)] * (N * 2)
    
    def de_callback(xk, convergence):
        return time.time() > deadline - 60  # Stop DE if we need time for local refinement
    
    try:
        de_result = differential_evolution(
            soft_min_obj,
            bounds,
            args=(50.0,),
            maxiter=100,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol',
            callback=de_callback
        )
        
        if de_result.success or de_result.fun < -np.inf:
            de_bary = clamp_bary(de_result.x.reshape(N, 2))
            update_best(de_bary)
            
    except Exception as e:
        pass

    # 2. Local Refinement of the DE result
    if best_bary is not None:
        if time.time() < deadline - 10:
            refined_de = refine_local(best_bary, time_limit=min(40.0, deadline - 10 - time.time()))
            update_best(refined_de)

    # 3. Fallback / Additional Search if DE didn't find anything good or time remains
    if time.time() < deadline - 30:
        # Try hexagonal initialization
        hex_bary = generate_hexagonal_init()
        refined_hex = refine_local(hex_bary, time_limit=min(20.0, deadline - 10 - time.time()))
        update_best(refined_hex)
        
        for seed in [1, 2, 3]:
            if time.time() > deadline - 10:
                break
            rng = np.random.default_rng(seed)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            rand_bary = np.column_stack([u, v])
            
            refined_rand = refine_local(rand_bary, time_limit=min(20.0, deadline - 10 - time.time()))
            update_best(refined_rand)

    # 4. Coordinated Ascent on best solution
    if best_bary is not None and time.time() < deadline - 15:
        ca_bary = coordinated_ascent(best_bary, time_limit=min(15.0, deadline - 10 - time.time()))
        update_best(ca_bary)

    # 5. Final Polish with very tight Nelder-Mead if time allows
    if best_bary is not None and time.time() < deadline - 5:
        final_polish = refine_local(best_bary, time_limit=min(5.0, deadline - 5 - time.time()))
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
