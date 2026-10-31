import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def symmetric_params_to_bary(params):
    """
    Convert symmetric parameters to barycentric coordinates.
    params: [r1, a1, r2, a2, r3, a3, axis_r, axis_a]
    - r1, r2, r3: radii of 3 orbits (distance from centroid in barycentric space)
    - a1, a2, a3: angular offsets for each orbit
    - axis_r: radius for the 11th point on a symmetry axis
    - axis_a: angle for the 11th point
    
    11 = 1 (centroid) + 3*3 (three orbits of 3) + 1 (axis point)
    """
    r1, a1, r2, a2, r3, a3, axis_r, axis_a = params
    
    centroid = np.array([1.0/3, 1.0/3])
    
    def orbit_point(r, a):
        # In barycentric coords, rotate around centroid
        # Reference direction: toward vertex 1 (u=1,v=0)
        # Barycentric rotation by angle theta around centroid
        # We use the fact that 120-degree rotation maps (u,v,w) -> (w,u,v)
        # For arbitrary angle, we interpolate in Cartesian and convert back
        cx, cy = 1.0/3, 1.0/3
        # Cartesian position at angle a from centroid at radius r
        # In barycentric, the "radial" direction toward vertex 1 is (2/3, -1/3, -1/3)
        # We'll work in Cartesian for the orbit
        # Centroid in Cartesian
        cart_cx = 0.5
        cart_cy = H / 3
        # Direction vector at angle a
        dx = r * np.cos(a)
        dy = r * np.sin(a)
        # But we need to map Cartesian offset to barycentric
        # A point at Cartesian (cart_cx + dx, cart_cy + dy)
        # Barycentric: u = 1 - x - y/H*... let me just compute directly
        x_cart = cart_cx + dx
        y_cart = cart_cy + dy
        # Convert to barycentric: x = v + 0.5*w, y = w*H
        # w = y/H, v = x - 0.5*w, u = 1 - v - w
        w_b = y_cart / H
        v_b = x_cart - 0.5 * w_b
        u_b = 1.0 - v_b - w_b
        return np.array([u_b, v_b])
    
    pts = [centroid.copy()]
    
    for r, a in [(r1, a1), (r2, a2), (r3, a3)]:
        # Generate 3 points at 120-degree intervals
        for k in range(3):
            angle = a + k * 2 * np.pi / 3
            pt = orbit_point(r, angle)
            pts.append(pt)
    
    # 11th point on a specific axis
    pt11 = orbit_point(axis_r, axis_a)
    pts.append(pt11)
    
    bary = np.array(pts)
    return clamp_bary(bary)

def symmetric_obj(params):
    """Objective for symmetric parameter optimization."""
    bary = symmetric_params_to_bary(params)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def symmetric_soft_obj(params, beta=30.0):
    """Soft min objective for symmetric parameters."""
    bary = symmetric_params_to_bary(params)
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

def generate_symmetric_init():
    """
    Generate a good symmetric configuration by optimizing the symmetric parameters.
    """
    # Initial guess: evenly spaced radii
    best_params = None
    best_val = -np.inf
    
    # Try several initial configurations
    init_configs = [
        [0.15, 0.0, 0.30, 0.0, 0.45, 0.0, 0.25, 0.0],
        [0.12, np.pi/6, 0.28, np.pi/6, 0.44, np.pi/6, 0.22, np.pi/3],
        [0.18, 0.0, 0.36, np.pi/3, 0.50, np.pi/3, 0.30, 0.0],
        [0.10, 0.0, 0.25, 0.0, 0.40, 0.0, 0.18, np.pi/6],
        [0.20, np.pi/12, 0.38, np.pi/12, 0.52, np.pi/12, 0.28, np.pi/12],
    ]
    
    bounds = [
        (0.05, 0.55), (0, 2*np.pi),  # r1, a1
        (0.05, 0.55), (0, 2*np.pi),  # r2, a2
        (0.05, 0.55), (0, 2*np.pi),  # r3, a3
        (0.05, 0.55), (0, 2*np.pi),  # axis_r, axis_a
    ]
    
    for init in init_configs:
        try:
            result = minimize(
                symmetric_soft_obj,
                init,
                args=(20.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            val = -symmetric_obj(result.x)
            if val > best_val:
                best_val = val
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([0.15, 0.0, 0.30, 0.0, 0.45, 0.0, 0.25, 0.0])
    
    bary = symmetric_params_to_bary(best_params)
    return bary, best_params

def generate_convex_hull_init():
    """
    Generate points near the convex hull (boundary) of the triangle.
    """
    rng = np.random.default_rng(42)
    pts = []
    
    # Place points along the three edges
    for edge in range(3):
        for t in [0.15, 0.35, 0.55, 0.75]:
            if edge == 0:  # bottom edge (0,0) to (1,0)
                pts.append([1-t, t])
            elif edge == 1:  # right edge (1,0) to (0.5, H)
                u = 1 - t
                v = t * 0.5
                pts.append([u, v])
            else:  # left edge (0.5, H) to (0,0)
                v = t
                u = 0
                pts.append([u, v])
    
    # Add centroid
    pts.append([1/3, 1/3])
    
    # Trim to 11 points if needed
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Add random interior points
        while len(pts) < N:
            u = rng.random()
            v = rng.random() * (1 - u)
            pts.append([u, v])
    
    bary = np.array(pts)
    return clamp_bary(bary)

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
    
    # Stage 1: Coarse soft-min (low beta)
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(15.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
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
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if result2.success or result2.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.4:
        return bary_current

    # Stage 3: Sharp soft-min (high beta)
    try:
        result3 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(150.0,),
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

    # Stage 4: Hard min refinement with Nelder-Mead
    try:
        result4 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result4.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.85:
        return bary_current

    # Stage 5: Final polish
    try:
        result5 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result5.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def refine_symmetric(params_init, time_limit=3.0):
    """
    Refine symmetric parameters using graduated soft-min.
    """
    start_time = time.time()
    bounds = [
        (0.03, 0.58), (0, 2*np.pi),
        (0.03, 0.58), (0, 2*np.pi),
        (0.03, 0.58), (0, 2*np.pi),
        (0.03, 0.58), (0, 2*np.pi),
    ]
    
    current = params_init.copy()
    
    # Stage 1: Low beta
    try:
        r1 = minimize(
            symmetric_soft_obj, current, args=(10.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if r1.fun < -symmetric_obj(current):
            current = r1.x.copy()
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.3:
        return current

    # Stage 2: Medium beta
    try:
        r2 = minimize(
            symmetric_soft_obj, current, args=(40.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-9}
        )
        if r2.fun < -symmetric_obj(current):
            current = r2.x.copy()
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.6:
        return current

    # Stage 3: High beta
    try:
        r3 = minimize(
            symmetric_soft_obj, current, args=(120.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if r3.fun < -symmetric_obj(current):
            current = r3.x.copy()
    except Exception:
        pass
    
    return current

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

    # Phase 1: Symmetric optimization (structural change)
    # This is the key innovation: optimize in the reduced symmetric space first
    try:
        sym_bary, sym_params = generate_symmetric_init()
        update_best(sym_bary)
        
        # Further refine the symmetric parameters
        remaining = deadline - time.time() - 20
        if remaining > 5:
            refined_params = refine_symmetric(sym_params, time_limit=min(5.0, remaining * 0.5))
            refined_sym_bary = symmetric_params_to_bary(refined_params)
            update_best(refined_sym_bary)
    except Exception:
        pass

    # Phase 2: Break symmetry and refine in full space
    if best_bary is not None:
        remaining = deadline - time.time() - 15
        if remaining > 5:
            # Refine the best symmetric solution in full space
            refined = refine_local(best_bary, time_limit=min(10.0, remaining))
            update_best(refined)

    # Phase 3: Try convex hull initialization
    if time.time() < deadline - 20:
        try:
            hull_bary = generate_convex_hull_init()
            remaining = deadline - time.time() - 10
            if remaining > 5:
                refined_hull = refine_local(hull_bary, time_limit=min(8.0, remaining))
                update_best(refined_hull)
        except Exception:
            pass

    # Phase 4: Try random starts
    if time.time() < deadline - 15:
        for seed in [42, 123, 456]:
            if time.time() > deadline - 10:
                break
            try:
                rand_bary = generate_random_init(seed=seed)
                remaining = deadline - time.time() - 5
                if remaining < 3:
                    break
                refined_rand = refine_local(rand_bary, time_limit=min(5.0, remaining))
                update_best(refined_rand)
            except Exception:
                continue

    # Phase 5: Perturb the best solution and refine
    if best_bary is not None and time.time() < deadline - 10:
        for sigma in [0.02, 0.04, 0.06]:
            if time.time() > deadline - 5:
                break
            try:
                perturbed = generate_perturbed_init(best_bary, sigma=sigma, seed=777)
                remaining = deadline - time.time() - 3
                if remaining < 3:
                    break
                refined_pert = refine_local(perturbed, time_limit=min(4.0, remaining))
                update_best(refined_pert)
            except Exception:
                continue

    # Phase 6: Final polish on the best solution
    if best_bary is not None and time.time() < deadline - 3:
        try:
            result_nm = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            update_best(clamp_bary(result_nm.x.reshape(N, 2)))
        except Exception:
            pass

    if best_bary is None:
        # Fallback
        best_bary, _ = generate_symmetric_init()

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
